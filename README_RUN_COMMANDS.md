# 常用运行命令（精简版）

```bash
# 0) 环境准备（Qwen2.5-Omni 推荐）
pip install -r requirement.txt
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview

# 1) GPT-5.2
python baseline/gpt52.py

# 2) GPT-4o-audio-preview
python baseline/gpt4o_audio_preview.py

# 3) Gemini 3.1 Flash Lite（preview）
python baseline/gemini3.py

# 4) Qwen2.5-Omni-7B（本地/服务器推理，首跑自动下载权重）
python baseline/qwen25.py

# 5) 四个新 baseline 的 smoke test（快速检查）
python baseline/gpt52.py --max-samples 50
python baseline/gpt4o_audio_preview.py --max-samples 50
python baseline/gemini3.py --max-samples 50
python baseline/qwen25.py --max-samples 20

# 6) 需要切换配置时（示例）
python baseline/gpt52.py --config yaml/gpt52_MERC.yaml
python baseline/qwen25.py --config yaml/qwen25_MERC.yaml

# 7) 旧 baseline 测试（Emotion / text）
python baseline/baseline.py -classify Emotion -modality text -test

# 8) 旧 baseline 测试（Emotion / audio）
python baseline/baseline.py -classify Emotion -modality audio -test

# 9) 旧 baseline 测试（Emotion / bimodal）
python baseline/baseline.py -classify Emotion -modality bimodal -test
```

