# MiniCPM-o 4.5 TTS DFlash

基于 MiniCPM-o 4.5 的语音推理项目，通过 DFlash 草稿模型加速 TTS codec token 解码，支持文本/图片/视频转语音、双工音视频交互及接受率评测。

## 安装

需要 Python 3.10、NVIDIA CUDA GPU、FFmpeg 和 uv。

```bash
uv sync
```

默认使用 `openbmb/MiniCPM-o-4_5` 和 `cs-qyzhang/MiniCPM-o-4_5-tts-dflash`，首次运行会下载模型。可用 `--model`、`--draft-model` 指定本地模型目录。

## 快速使用

```bash
# 文本转语音
uv run minicpmo-dflash-simplex --text "你好，欢迎使用语音合成。" --output outputs/demo.wav

# 双工音视频交互：请自行提供输入视频
uv run minicpmo-dflash-duplex --video /path/to/input.mp4

# DFlash 接受率评测
uv run minicpmo-dflash-benchmark --output outputs/benchmark.json
```

常用参数：`--ref-audio` 指定参考音色，`--device cuda:0` 选择 GPU，`--vocoder-device cuda:1` 将声码器放到另一张显卡，`--disable-dflash` 切换为原生 TTS 解码。各命令的完整参数可通过 `--help` 查看。

DFlash 仅加速 TTS 解码；主模型的文本生成与视觉、音频编码保持原有流程。生成文件默认保存在 `outputs/`。
