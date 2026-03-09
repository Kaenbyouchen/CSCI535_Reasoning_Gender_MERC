# CSCI535 MERC 实验说明（CARC 版）

本 README 面向当前项目的实际使用流程：  
本地开发 -> 推送 GitHub -> CARC 拉取代码 -> 手动上传大数据 -> 运行评测。

仓库地址：[`Kaenbyouchen/CSCI535_Reasoning_Gender_MERC`](https://github.com/Kaenbyouchen/CSCI535_Reasoning_Gender_MERC)

---

## 1. 项目结构（与运行强相关）

- `baseline/`：评测脚本（`gpt52.py`、`gpt4o_audio_preview.py`、`gemini3.py`、`qwen25.py`）
- `yaml/`：模型配置（包含路径、模态、prompt、推理粒度）
- `data/`：标注与数据集目录（如 `MELD`、`IEMOCAP`）
- `MELD.Raw/`：MELD 原始视频（大文件，不进 Git）
- `result/`：评测输出目录（大文件，不进 Git）

---

## 2. 首次推送到 GitHub（本地）

在项目根目录执行：

```bash
git init
git add .
git commit -m "Initial commit: MERC baselines and configs"
git branch -M main
git remote add origin https://github.com/Kaenbyouchen/CSCI535_Reasoning_Gender_MERC.git
git push -u origin main
```

后续代码更新：

```bash
git add .
git commit -m "update: <message>"
git push
```

---

## 3. CARC 端首次部署

### 3.1 拉取代码

```bash
cd ~
git clone https://github.com/Kaenbyouchen/CSCI535_Reasoning_Gender_MERC.git
cd CSCI535_Reasoning_Gender_MERC
```

### 3.2 配置 Python 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirement.txt
```

### 3.3 Qwen2.5-Omni 推荐 transformers 版本

```bash
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install accelerate
```

### 3.4 检查 ffmpeg（音视频评测必需）

```bash
which ffmpeg
ffmpeg -version
```

---

## 4. 数据同步策略（推荐）

由于数据与结果体积大，建议：

- 代码走 Git（push/pull）
- 大文件手动上传/rsync 到 CARC（不进 Git）

建议保持下列路径在 CARC 与本地一致：

- `MELD.Raw/`
- `data/IEMOCAP/`
- 其他你需要的 `data/pickles/`、`data/models/`

---

## 5. 在 CARC 运行评测

激活环境后执行：

```bash
source .venv/bin/activate
```

### Qwen2.5-Omni-7B（本地权重推理，首跑自动下载）

```bash
python baseline/qwen25.py --config yaml/qwen25_MERC.yaml
```

快速烟雾测试：

```bash
python baseline/qwen25.py --max-samples 20
```

### Gemini / GPT 基线

```bash
python baseline/gemini3.py
python baseline/gpt52.py
python baseline/gpt4o_audio_preview.py
```

---

## 6. 输出文件说明

每次运行会在 `result/` 下生成新的 run 目录，通常包含：

- `run_config.yaml`：该次运行实际配置快照
- `predictions_detailed.csv`：逐样本预测明细
- `metrics.json`：总体和分组指标

此外会更新：

- `result/result_summary.json`：多次运行的汇总记录

---

## 7. 常见问题

### 7.1 `ffmpeg` 找不到

- 先确认 `ffmpeg -version` 是否可执行
- CARC 可用 module 或 conda 安装后再运行

### 7.2 Qwen 报 `KeyError: 'qwen2_5_omni'`

- 通常是 `transformers` 版本不对
- 按第 3.3 节重装 preview 版本

### 7.3 首次下载模型很慢

- 属于正常现象（会下载到 Hugging Face 缓存）
- 后续重复运行通常不会再次完整下载

