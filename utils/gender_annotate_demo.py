import argparse
from collections import Counter, defaultdict
import subprocess
import sys
from pathlib import Path

from gender_annotate import (
    HIGH_RISK_GENERIC,
    apply_gender_by_dialogue,
    apply_gender_by_speaker,
    clear_all_gender,
    ensure_gender_field,
    read_csv_rows,
    should_skip_speaker,
    speaker_status,
    unit_status,
)


class DemoQuickTimePlayer:
    def __init__(self):
        self.mode = None

    def stop(self):
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
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", "QuickTime Player", path], check=False)
            self.mode = "quicktime"
            return
        print(f"[WARN] QuickTime is only available on macOS. Please open manually: {path}")


class DemoClipResolver:
    def __init__(self, clips_dir):
        self.clips_dir = Path(clips_dir)
        self.index = {}
        self.audio_cache = {}
        self._index_local_mp4()

    def _index_local_mp4(self):
        if not self.clips_dir.exists():
            return
        for p in self.clips_dir.rglob("*.mp4"):
            name = p.name
            if name.startswith("._"):
                continue
            if name not in self.index:
                self.index[name] = p

    def resolve(self, _split, dialogue_id, utterance_id):
        base_name = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        local = self.index.get(base_name)
        if local and local.exists():
            return local
        return None

    def has_audio_stream(self, video_path):
        if not video_path:
            return False
        key = str(video_path)
        if key in self.audio_cache:
            return self.audio_cache[key]
        # ffprobe: any audio stream => output "audio"
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            key,
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            has_audio = "audio" in out.lower()
        except Exception:
            has_audio = False
        self.audio_cache[key] = has_audio
        return has_audio


def ask_gender_demo():
    while True:
        ans = input("Enter gender [1=male, 2=female, s=skip, q=quit]: ").strip().lower()
        if ans == "1":
            return "male"
        if ans == "2":
            return "female"
        if ans in {"s", "q"}:
            return ans
        print("Invalid input. Please enter 1/2/s/q.")


def collect_test_only_data(meld_data_dir):
    csv_path = Path(meld_data_dir) / "test_sent_emo.csv"
    rows, fieldnames = read_csv_rows(csv_path)

    split_rows = {"test": rows}
    split_fields = {"test": fieldnames}
    speaker_counter = Counter()
    speaker_dialogues = defaultdict(set)
    speaker_first = {}
    speaker_dialogue_first = {}

    for row in rows:
        speaker = str(row.get("Speaker", "")).strip()
        dialogue_id = str(row.get("Dialogue_ID", ""))
        utterance_id = str(row.get("Utterance_ID", ""))
        utterance = str(row.get("Utterance", ""))
        split = "test"

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


def is_skipped_clip_name(clip_path):
    if not clip_path:
        return False
    # Known bad clip for demo (no audio track).
    return Path(clip_path).name in {"dia16_utt3.mp4"}


def main():
    parser = argparse.ArgumentParser(
        description="DEMO mode: follow gender_annotate.py workflow but never save to CSV."
    )
    parser.add_argument("--meld-data-dir", default="data/MELD", help="MELD CSV directory")
    parser.add_argument("--raw-dir", default="MELD.Raw", help="MELD raw directory root")
    parser.add_argument(
        "--clips-dir",
        default="",
        help="Directory containing extracted test clips (default: MELD.Raw/output_repeated_splits_test)",
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all existing Gender labels at startup (memory only, no file write)",
    )
    parser.add_argument(
        "--play-all",
        action="store_true",
        help="Ignore resume status and play/annotate all units for demo recording",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only keep clips that contain an audio stream",
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
    ) = collect_test_only_data(meld_data_dir)
    ensure_gender_field(split_fields)
    if args.clear_all:
        clear_all_gender(split_rows)
        print("All existing Gender labels cleared in memory (not persisted).")

    all_speakers = sorted(speaker_counter.keys())
    skipped_speakers = sorted([s for s in all_speakers if should_skip_speaker(s)])
    eligible_speakers = [s for s in all_speakers if not should_skip_speaker(s)]

    type1_speakers = sorted(
        [
            s
            for s in eligible_speakers
            if (speaker_counter[s] == 1) or (s not in HIGH_RISK_GENERIC)
        ]
    )

    type2_speakers = sorted([s for s in eligible_speakers if s not in type1_speakers])
    type2_units = []
    for speaker in type2_speakers:
        for split, dialogue_id in sorted(speaker_dialogues[speaker]):
            type2_units.append((speaker, split, dialogue_id))

    type1_done = sum(1 for s in type1_speakers if speaker_status(split_rows, s) == "done")
    type1_conflict = sum(
        1 for s in type1_speakers if speaker_status(split_rows, s) == "conflict"
    )
    type1_todo = [
        s
        for s in type1_speakers
        if speaker_status(split_rows, s) != "done"
    ]

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

    if args.play_all:
        type1_todo = list(type1_speakers)
        type2_todo = list(type2_units)

    print(f"Total speakers: {len(all_speakers)}")
    print(f"Skipped speakers (rule-based): {len(skipped_speakers)}")
    print(f"Type-1 units (single annotation per speaker): {len(type1_speakers)}")
    print(
        f"Type-2 units (high-risk generic names by dialogue): {len(type2_units)} "
        f"(from {len(type2_speakers)} speakers)"
    )
    print(f"Total annotation actions: {len(type1_speakers) + len(type2_units)}")
    print("Demo split filter: test (test_sent_emo.csv only)")
    print(
        f"Resume status -> Type-1: done {type1_done}/{len(type1_speakers)}, "
        f"todo {len(type1_todo)}, conflict {type1_conflict}"
    )
    print(
        f"Resume status -> Type-2: done {type2_done}/{len(type2_units)}, "
        f"todo {len(type2_todo)}, conflict {type2_conflict}"
    )
    if args.play_all:
        print("[DEMO] play-all mode enabled: resume status is ignored.")
    clips_dir = args.clips_dir or str(raw_dir / "output_repeated_splits_test")
    print(f"[DEMO] Clip source: {clips_dir}")
    print("[DEMO] This script only reads clips from the extracted test folder.")
    print("[DEMO] This script will NOT write anything back to CSV files.")

    if not type1_todo and not type2_todo:
        print(
            "[DEMO] No pending items detected under resume logic. "
            "Run with --play-all or --clear-all to force playback."
        )
        return

    resolver = DemoClipResolver(clips_dir)
    player = DemoQuickTimePlayer()

    def sort_type1_by_audio(speakers):
        items = []
        for spk in speakers:
            meta = speaker_first[spk]
            vp = resolver.resolve(meta["split"], meta["Dialogue_ID"], meta["Utterance_ID"])
            items.append((spk, vp, resolver.has_audio_stream(vp)))
        # audio first, then stable speaker order
        items.sort(key=lambda x: (not x[2], x[0]))
        return [x[0] for x in items], sum(1 for x in items if x[2])

    def sort_type2_by_audio(units):
        items = []
        for spk, split, dialogue_id in units:
            meta = speaker_dialogue_first[(spk, split, dialogue_id)]
            vp = resolver.resolve(split, dialogue_id, meta["Utterance_ID"])
            items.append(((spk, split, dialogue_id), vp, resolver.has_audio_stream(vp)))
        items.sort(key=lambda x: (not x[2], x[0][0], x[0][1], x[0][2]))
        return [x[0] for x in items], sum(1 for x in items if x[2])

    type1_todo, type1_audio_n = sort_type1_by_audio(type1_todo)
    type2_todo, type2_audio_n = sort_type2_by_audio(type2_todo)

    if args.audio_only:
        before_t1, before_t2 = len(type1_todo), len(type2_todo)
        type1_todo = [
            spk
            for spk in type1_todo
            if resolver.has_audio_stream(
                resolver.resolve(
                    speaker_first[spk]["split"],
                    speaker_first[spk]["Dialogue_ID"],
                    speaker_first[spk]["Utterance_ID"],
                )
            )
            and not is_skipped_clip_name(
                resolver.resolve(
                    speaker_first[spk]["split"],
                    speaker_first[spk]["Dialogue_ID"],
                    speaker_first[spk]["Utterance_ID"],
                )
            )
        ]
        type2_todo = [
            (spk, split, dialogue_id)
            for spk, split, dialogue_id in type2_todo
            if resolver.has_audio_stream(
                resolver.resolve(
                    split,
                    dialogue_id,
                    speaker_dialogue_first[(spk, split, dialogue_id)]["Utterance_ID"],
                )
            )
            and not is_skipped_clip_name(
                resolver.resolve(
                    split,
                    dialogue_id,
                    speaker_dialogue_first[(spk, split, dialogue_id)]["Utterance_ID"],
                )
            )
        ]
        print(
            f"[DEMO] audio-only mode enabled: "
            f"Type-1 kept {len(type1_todo)}/{before_t1}, "
            f"Type-2 kept {len(type2_todo)}/{before_t2}"
        )
    else:
        # Even without audio-only, skip known problematic clips.
        type1_todo = [
            spk
            for spk in type1_todo
            if not is_skipped_clip_name(
                resolver.resolve(
                    speaker_first[spk]["split"],
                    speaker_first[spk]["Dialogue_ID"],
                    speaker_first[spk]["Utterance_ID"],
                )
            )
        ]
        type2_todo = [
            (spk, split, dialogue_id)
            for spk, split, dialogue_id in type2_todo
            if not is_skipped_clip_name(
                resolver.resolve(
                    split,
                    dialogue_id,
                    speaker_dialogue_first[(spk, split, dialogue_id)]["Utterance_ID"],
                )
            )
        ]

    print(
        f"[DEMO] Audio-priority order applied: "
        f"Type-1 audio={type1_audio_n}/{len(type1_todo)}, "
        f"Type-2 audio={type2_audio_n}/{len(type2_todo)}"
    )

    completed_type1 = 0
    completed_type2 = 0
    try:
        completed_type1_total = type1_done
        for speaker in type1_todo:
            player.stop()
            meta = speaker_first[speaker]
            split = meta["split"]
            dia = meta["Dialogue_ID"]
            utt = meta["Utterance_ID"]
            utt_text = meta["Utterance"]

            print("\n" + "=" * 80)
            print("[Type-1] One annotation per speaker")
            print(f"Progress: {completed_type1_total + 1}/{len(type1_speakers)}")
            print(f"Speaker: {speaker}")
            print(
                f"Stats: utt={speaker_counter[speaker]}, dialogues={len(speaker_dialogues[speaker])}"
            )
            print(f"Sample: {split} / dia{dia}_utt{utt}")
            print(f"Utterance: {utt_text}")

            video = resolver.resolve(split, dia, utt)
            if video:
                print(f"Playing clip: {video}")
                player.play(video)
            else:
                print("[WARN] Clip not found. Please label from text only.")

            result = ask_gender_demo()
            if result == "s":
                print(f"Skipped: {speaker}")
                continue
            if result == "q":
                print("Quit received. Leaving DEMO without saving CSV.")
                player.stop()
                break

            apply_gender_by_speaker(split_rows, speaker, result)
            completed_type1 += 1
            completed_type1_total += 1
            print(f"Labeled: {speaker} -> {result}")

        completed_type2_total = type2_done
        for speaker, split, dialogue_id in type2_todo:
            player.stop()
            meta = speaker_dialogue_first[(speaker, split, dialogue_id)]
            utt = meta["Utterance_ID"]
            utt_text = meta["Utterance"]

            print("\n" + "=" * 80)
            print("[Type-2] High-risk generic names by dialogue")
            print(f"Progress: {completed_type2_total + 1}/{len(type2_units)}")
            print(f"Speaker: {speaker}")
            print(f"Unit: {split} / dialogue={dialogue_id}")
            print(f"Sample: dia{dialogue_id}_utt{utt}")
            print(f"Utterance: {utt_text}")

            video = resolver.resolve(split, dialogue_id, utt)
            if video:
                print(f"Playing clip: {video}")
                player.play(video)
            else:
                print("[WARN] Clip not found. Please label from text only.")

            result = ask_gender_demo()
            if result == "s":
                print(f"Skipped: {speaker} @ {split}/dia{dialogue_id}")
                continue
            if result == "q":
                print("Quit received. Leaving DEMO without saving CSV.")
                player.stop()
                break

            apply_gender_by_dialogue(split_rows, speaker, split, dialogue_id, result)
            completed_type2 += 1
            completed_type2_total += 1
            print(f"Labeled: {speaker} @ {split}/dia{dialogue_id} -> {result}")

    except KeyboardInterrupt:
        print("\nInterrupted. Leaving DEMO without saving CSV.")
    finally:
        player.stop()

    print(f"\n[DEMO] No files were written under: {meld_data_dir}")
    print(f"[DEMO] Type-1 completed this run: {completed_type1}")
    print(f"[DEMO] Type-2 completed this run: {completed_type2}")
    print(f"[DEMO] Total completed this run: {completed_type1 + completed_type2}")


if __name__ == "__main__":
    main()
