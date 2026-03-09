import argparse
import csv
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


CSV_FILES = [
    ("train", "train_sent_emo.csv"),
    ("dev", "dev_sent_emo.csv"),
    ("test", "test_sent_emo.csv"),
]

TAR_FILES = {
    "train": "train.tar.gz",
    "dev": "dev.tar.gz",
    "test": "test.tar.gz",
}

# 高风险泛称：这组按 split+dialogue 标注
HIGH_RISK_GENERIC = {
    "All",
    "Customer",
    "Director",
    "Doctor",
    "Fireman No. 3",
    "Flight Attendant",
    "Girl",
    "Guy",
    "Hitchhiker",
    "Hold Voice",
    "Joey/Drake",
    "Man",
    "Nurse",
    "Policeman",
    "Receptionist",
    "Stage Director",
    "Student",
    "The Casting Director",
    "The Cooking Teacher",
    "The Director",
    "The Dry Cleaner",
    "The Head Librarian",
    "The Instructor",
    "The Interviewer",
    "The Museum Official",
    "The Potential Roommate",
    "Ticket Counter Attendant",
    "Tour Guide",
    "Waitress",
    "Woman",
}


def should_skip_speaker(speaker):
    s = (speaker or "").strip()
    if not s:
        return True
    s_lower = s.lower()

    # 1) A and B 组合
    if re.search(r"\band\b", s_lower):
        return True
    # 2) 斜杠组合
    if "/" in s:
        return True
    # 3) 群体说话人
    if s_lower in {"all", "both"}:
        return True
    return False


def read_csv_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv_rows(csv_path, rows, fieldnames):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_gender(raw):
    v = (raw or "").strip().lower()
    if v in {"male", "m", "1", "男", "男生"}:
        return "male"
    if v in {"female", "f", "2", "女", "女生"}:
        return "female"
    return ""


def open_video(video_path):
    if sys.platform == "darwin":
        subprocess.run(["open", str(video_path)], check=False)
    elif shutil.which("xdg-open"):
        subprocess.run(["xdg-open", str(video_path)], check=False)
    else:
        print(f"[WARN] 无法自动打开视频，请手动查看: {video_path}")


class VideoPlayer:
    def __init__(self):
        self.proc = None
        self.mode = None

    def stop(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None
            self.mode = None
            return

        # macOS fallback: 使用 QuickTime 打开时，尝试关闭当前文档
        if self.mode == "quicktime":
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "QuickTime Player" to if (count of documents) > 0 then close front document',
                ],
                check=False,
            )
            self.mode = None

    def play(self, video_path):
        self.stop()
        path = str(video_path)

        # 优先使用可控播放器进程，方便切换时自动关闭
        if shutil.which("ffplay"):
            self.proc = subprocess.Popen(
                ["ffplay", "-autoexit", "-loglevel", "quiet", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.mode = "process"
            return
        if shutil.which("mpv"):
            self.proc = subprocess.Popen(
                ["mpv", "--force-window=yes", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.mode = "process"
            return
        if shutil.which("vlc"):
            self.proc = subprocess.Popen(
                ["vlc", "--play-and-exit", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.mode = "process"
            return

        # fallback: 指定 QuickTime，后续用 AppleScript 关掉上一个
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", "QuickTime Player", path], check=False)
            self.mode = "quicktime"
            return

        open_video(path)
        self.mode = "unmanaged"


class VideoResolver:
    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="meld_gender_"))
        self.extracted_map = {}
        self.tar_indexes = {}

        self._build_extracted_index()

    def _build_extracted_index(self):
        # 如果用户已经提前解压了视频，这里可直接定位。
        for p in self.raw_dir.rglob("*.mp4"):
            name = p.name
            if name.startswith("._"):
                continue
            if name not in self.extracted_map:
                self.extracted_map[name] = p

    def _build_tar_index(self, split):
        if split in self.tar_indexes:
            return
        tar_path = self.raw_dir / TAR_FILES[split]
        index = {}
        if not tar_path.exists():
            self.tar_indexes[split] = (None, index)
            return

        tar_obj = tarfile.open(tar_path, "r:gz")
        for member in tar_obj.getmembers():
            if not member.isfile():
                continue
            if not member.name.lower().endswith(".mp4"):
                continue
            base = Path(member.name).name
            if base.startswith("._"):
                continue
            if base not in index:
                index[base] = member.name
        self.tar_indexes[split] = (tar_obj, index)

    def resolve(self, split, dialogue_id, utterance_id):
        base_name = f"dia{dialogue_id}_utt{utterance_id}.mp4"

        # 先找已解压的本地文件
        local = self.extracted_map.get(base_name)
        if local and local.exists():
            return local

        # 再从 tar 包里提取一个临时文件
        self._build_tar_index(split)
        tar_obj, index = self.tar_indexes.get(split, (None, {}))
        member_name = index.get(base_name)
        if tar_obj and member_name:
            member = tar_obj.getmember(member_name)
            extracted = self.temp_dir / base_name
            if not extracted.exists():
                fobj = tar_obj.extractfile(member)
                if fobj is None:
                    return None
                with extracted.open("wb") as out:
                    out.write(fobj.read())
            return extracted
        return None

    def preload_targets(self, targets_by_split):
        """
        预提取目标视频，避免交互阶段每次切换都等待 tar.gz 解压。
        targets_by_split: {"train": {"dia0_utt0.mp4", ...}, "dev": {...}, "test": {...}}
        """
        for split, targets in targets_by_split.items():
            if not targets:
                continue

            # 已有本地解压文件则跳过
            pending = {name for name in targets if name not in self.extracted_map}
            if not pending:
                continue

            tar_path = self.raw_dir / TAR_FILES[split]
            if not tar_path.exists():
                print(f"[WARN] 未找到 {tar_path}，跳过 {split} 预提取。")
                continue

            print(f"[预提取] {split}: 目标 {len(pending)} 个视频，正在缓存...")
            hit = 0
            with tarfile.open(tar_path, "r:gz") as tar_obj:
                for member in tar_obj:
                    if not member.isfile():
                        continue
                    if not member.name.lower().endswith(".mp4"):
                        continue
                    base = Path(member.name).name
                    if base.startswith("._"):
                        continue
                    if base not in pending:
                        continue

                    dst = self.temp_dir / base
                    if not dst.exists():
                        fobj = tar_obj.extractfile(member)
                        if fobj is None:
                            continue
                        with dst.open("wb") as out:
                            out.write(fobj.read())
                    self.extracted_map[base] = dst
                    hit += 1
                    pending.remove(base)
                    if not pending:
                        break

            if pending:
                print(f"[预提取] {split}: 完成 {hit} 个，缺失 {len(pending)} 个（将回退到按需提取）。")
            else:
                print(f"[预提取] {split}: 完成 {hit} 个。")


def collect_data(meld_data_dir):
    split_rows = {}
    split_fields = {}
    speaker_counter = Counter()
    speaker_dialogues = defaultdict(set)
    speaker_first = {}
    speaker_dialogue_first = {}

    for split, filename in CSV_FILES:
        csv_path = meld_data_dir / filename
        rows, fieldnames = read_csv_rows(csv_path)

        split_rows[split] = rows
        split_fields[split] = fieldnames

        for row in rows:
            speaker = row["Speaker"].strip()
            dialogue_id = row["Dialogue_ID"]
            utterance_id = row["Utterance_ID"]
            utterance = row["Utterance"]

            speaker_counter[speaker] += 1
            speaker_dialogues[speaker].add((split, dialogue_id))

            if speaker not in speaker_first:
                speaker_first[speaker] = {
                    "split": split,
                    "Dialogue_ID": dialogue_id,
                    "Utterance_ID": utterance_id,
                    "Utterance": utterance,
                }

            key = (speaker, split, dialogue_id)
            if key not in speaker_dialogue_first:
                speaker_dialogue_first[key] = {
                    "split": split,
                    "Dialogue_ID": dialogue_id,
                    "Utterance_ID": utterance_id,
                    "Utterance": utterance,
                }

    return (
        split_rows,
        split_fields,
        speaker_counter,
        speaker_dialogues,
        speaker_first,
        speaker_dialogue_first,
    )


def apply_gender_by_speaker(split_rows, speaker, gender):
    for rows in split_rows.values():
        for row in rows:
            if row["Speaker"].strip() == speaker:
                row["Gender"] = gender


def apply_gender_by_dialogue(split_rows, speaker, split, dialogue_id, gender):
    rows = split_rows[split]
    for row in rows:
        if row["Speaker"].strip() == speaker and row["Dialogue_ID"] == dialogue_id:
            row["Gender"] = gender


def clear_all_gender(split_rows):
    for rows in split_rows.values():
        for row in rows:
            row["Gender"] = ""


def speaker_status(split_rows, speaker):
    seen_any = False
    genders = set()
    for rows in split_rows.values():
        for row in rows:
            if row["Speaker"].strip() != speaker:
                continue
            seen_any = True
            g = normalize_gender(row.get("Gender", ""))
            if not g:
                return "unresolved"
            genders.add(g)
            if len(genders) > 1:
                return "conflict"
    if not seen_any:
        return "unresolved"
    return "done"


def unit_status(split_rows, speaker, split, dialogue_id):
    seen_any = False
    genders = set()
    for row in split_rows[split]:
        if row["Speaker"].strip() != speaker or row["Dialogue_ID"] != dialogue_id:
            continue
        seen_any = True
        g = normalize_gender(row.get("Gender", ""))
        if not g:
            return "unresolved"
        genders.add(g)
        if len(genders) > 1:
            return "conflict"
    if not seen_any:
        return "unresolved"
    return "done"


def ensure_gender_field(split_fields):
    for split in split_fields:
        if "Gender" not in split_fields[split]:
            split_fields[split].append("Gender")


def save_all(meld_data_dir, split_rows, split_fields):
    for split, filename in CSV_FILES:
        csv_path = meld_data_dir / filename
        write_csv_rows(csv_path, split_rows[split], split_fields[split])


def ask_gender():
    while True:
        ans = input("请输入性别 [1=男, 2=女, s=跳过, q=保存并退出]: ").strip().lower()
        if ans == "1":
            return "male"
        if ans == "2":
            return "female"
        if ans in {"s", "q"}:
            return ans
        print("输入无效，请输入 1/2/s/q。")


def main():
    parser = argparse.ArgumentParser(description="为 MELD 标注 speaker 性别并写回 CSV。")
    parser.add_argument("--meld-data-dir", default="data/MELD", help="MELD 标注 csv 目录")
    parser.add_argument("--raw-dir", default="MELD.Raw", help="MELD 原始视频目录（包含 train/dev/test.tar.gz 或已解压视频）")
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="启动时清空全部已有 Gender（默认断点续标）",
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="不预提取待标注视频（默认会预提取以提升播放速度）",
    )
    args = parser.parse_args()

    meld_data_dir = Path(args.meld_data_dir)
    raw_dir = Path(args.raw_dir)

    (
        split_rows,
        split_fields,
        speaker_counter,
        speaker_dialogues,
        speaker_first,
        speaker_dialogue_first,
    ) = collect_data(meld_data_dir)
    ensure_gender_field(split_fields)
    if args.clear_all:
        clear_all_gender(split_rows)
        print("已清空全部现有 Gender 标注。")

    all_speakers = sorted(speaker_counter.keys())
    skipped_speakers = sorted([s for s in all_speakers if should_skip_speaker(s)])
    eligible_speakers = [s for s in all_speakers if not should_skip_speaker(s)]

    # 第一类：utt=1 或 非高风险泛称 => 按 speaker 一次
    type1_speakers = sorted(
        [
            s
            for s in eligible_speakers
            if (speaker_counter[s] == 1) or (s not in HIGH_RISK_GENERIC)
        ]
    )

    # 第二类：剩余（高风险泛称且 utt>1）=> 按 split+dialogue 标
    type2_speakers = sorted([s for s in eligible_speakers if s not in type1_speakers])
    type2_units = []
    for speaker in type2_speakers:
        for split, dialogue_id in sorted(speaker_dialogues[speaker]):
            type2_units.append((speaker, split, dialogue_id))

    type1_done = sum(1 for s in type1_speakers if speaker_status(split_rows, s) == "done")
    type1_conflict = sum(
        1 for s in type1_speakers if speaker_status(split_rows, s) == "conflict"
    )
    type1_todo = [s for s in type1_speakers if speaker_status(split_rows, s) != "done"]

    type2_done = sum(
        1 for s, split, d in type2_units if unit_status(split_rows, s, split, d) == "done"
    )
    type2_conflict = sum(
        1
        for s, split, d in type2_units
        if unit_status(split_rows, s, split, d) == "conflict"
    )
    type2_todo = [
        (s, split, d)
        for s, split, d in type2_units
        if unit_status(split_rows, s, split, d) != "done"
    ]

    print(f"总 speaker 数: {len(all_speakers)}")
    print(f"跳过标注 speaker 数: {len(skipped_speakers)}")
    print(f"第一类（按 speaker 一次）: {len(type1_speakers)}")
    print(
        f"第二类（高风险泛称按 dialogue）: {len(type2_units)} "
        f"(来自 {len(type2_speakers)} 个 speaker)"
    )
    print(f"总标注次数: {len(type1_speakers) + len(type2_units)}")
    print(
        f"断点续标状态 -> 第一类: 已完成 {type1_done}/{len(type1_speakers)}, "
        f"待标注 {len(type1_todo)}, 冲突 {type1_conflict}"
    )
    print(
        f"断点续标状态 -> 第二类: 已完成 {type2_done}/{len(type2_units)}, "
        f"待标注 {len(type2_todo)}, 冲突 {type2_conflict}"
    )

    resolver = VideoResolver(raw_dir)
    player = VideoPlayer()

    if not args.no_preload:
        targets_by_split = {"train": set(), "dev": set(), "test": set()}
        for speaker in type1_todo:
            meta = speaker_first[speaker]
            targets_by_split[meta["split"]].add(
                f"dia{meta['Dialogue_ID']}_utt{meta['Utterance_ID']}.mp4"
            )
        for speaker, split, dialogue_id in type2_todo:
            meta = speaker_dialogue_first[(speaker, split, dialogue_id)]
            targets_by_split[split].add(
                f"dia{meta['Dialogue_ID']}_utt{meta['Utterance_ID']}.mp4"
            )
        resolver.preload_targets(targets_by_split)

    completed_type1 = 0
    completed_type2 = 0
    try:
        # 第一阶段：按 speaker 标注
        completed_type1_total = type1_done
        for speaker in type1_todo:
            player.stop()
            meta = speaker_first[speaker]
            split = meta["split"]
            dia = meta["Dialogue_ID"]
            utt = meta["Utterance_ID"]
            utt_text = meta["Utterance"]

            print("\n" + "=" * 80)
            print("[第一类] 按 speaker 一次")
            print(f"进度: {completed_type1_total + 1}/{len(type1_speakers)}")
            print(f"Speaker: {speaker}")
            print(
                f"统计: utt={speaker_counter[speaker]}, dialogues={len(speaker_dialogues[speaker])}"
            )
            print(f"样例: {split} / dia{dia}_utt{utt}")
            print(f"Utterance: {utt_text}")

            video = resolver.resolve(split, dia, utt)
            if video:
                print(f"播放视频: {video}")
                player.play(video)
            else:
                print("[WARN] 未找到对应视频，将仅显示文本供你判断。")

            result = ask_gender()
            if result == "s":
                print(f"已跳过: {speaker}")
                continue
            if result == "q":
                print("收到退出指令，正在保存当前进度...")
                player.stop()
                break

            apply_gender_by_speaker(split_rows, speaker, result)
            completed_type1 += 1
            completed_type1_total += 1
            print(f"已标注: {speaker} -> {result}")

        # 第二阶段：高风险泛称，按 split+dialogue 标注
        completed_type2_total = type2_done
        for speaker, split, dialogue_id in type2_todo:
            player.stop()
            meta = speaker_dialogue_first[(speaker, split, dialogue_id)]
            utt = meta["Utterance_ID"]
            utt_text = meta["Utterance"]

            print("\n" + "=" * 80)
            print("[第二类] 高风险泛称按 dialogue 标注")
            print(f"进度: {completed_type2_total + 1}/{len(type2_units)}")
            print(f"Speaker: {speaker}")
            print(f"单元: {split} / dialogue={dialogue_id}")
            print(f"样例: dia{dialogue_id}_utt{utt}")
            print(f"Utterance: {utt_text}")

            video = resolver.resolve(split, dialogue_id, utt)
            if video:
                print(f"播放视频: {video}")
                player.play(video)
            else:
                print("[WARN] 未找到对应视频，将仅显示文本供你判断。")

            result = ask_gender()
            if result == "s":
                print(f"已跳过: {speaker} @ {split}/dia{dialogue_id}")
                continue
            if result == "q":
                print("收到退出指令，正在保存当前进度...")
                player.stop()
                break

            apply_gender_by_dialogue(split_rows, speaker, split, dialogue_id, result)
            completed_type2 += 1
            completed_type2_total += 1
            print(f"已标注: {speaker} @ {split}/dia{dialogue_id} -> {result}")

    except KeyboardInterrupt:
        print("\n检测到中断，正在保存当前进度...")
    finally:
        player.stop()

    save_all(meld_data_dir, split_rows, split_fields)
    print(f"\n已保存到: {meld_data_dir}")
    print(f"本次第一类完成: {completed_type1}")
    print(f"本次第二类完成: {completed_type2}")
    print(f"本次总完成: {completed_type1 + completed_type2}")


if __name__ == "__main__":
    main()
