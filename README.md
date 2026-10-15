# Speculative Decoding：DFlash 语音推理实验

本仓库用于个人运行 DFlash 的 MiniCPM-o 4.5 TTS 推理实验，支持文本/图片/视频转语音、双工音视频交互及接受率评测。方法参考 [DFlash 官方论文](https://arxiv.org/abs/2602.06036)：用轻量级块扩散模型并行生成草稿，再由目标模型验证。

## 安装

需要 Python 3.10、NVIDIA CUDA GPU、FFmpeg 和 uv。

```bash
uv sync
```

基础模型使用 `openbmb/MiniCPM-o-4_5`，首次运行会下载模型，也可用 `--model` 指定本地目录。以下示例需将 `/path/to/tts-draft` 替换为兼容 MiniCPM-o Talker 的本地 DFlash 草稿权重目录，包含本项目加载器所需的 `config.json`、`model.safetensors` 和词表映射。

官方文本模型草稿权重见下方资源链接；它们面向对应的目标模型，不能直接替代本实验的 TTS 草稿权重。

## 快速使用

```bash
# 文本转语音
uv run minicpmo-dflash-simplex --draft-model /path/to/tts-draft --text "你好，欢迎使用语音合成。" --output outputs/demo.wav

# 双工音视频交互：请自行提供输入视频
uv run minicpmo-dflash-duplex --draft-model /path/to/tts-draft --video /path/to/input.mp4

# DFlash 接受率评测
uv run minicpmo-dflash-benchmark --draft-model /path/to/tts-draft --output outputs/benchmark.json
```

常用参数：`--ref-audio` 指定参考音色，`--device cuda:0` 选择 GPU，`--vocoder-device cuda:1` 将声码器放到另一张显卡，`--disable-dflash` 切换为原生 TTS 解码。各命令的完整参数可通过 `--help` 查看。

DFlash 仅加速 TTS 解码；主模型的文本生成与视觉、音频编码保持原有流程。生成文件默认保存在 `outputs/`。

## DFlash 官方参考

- 论文：[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)（Jian Chen、Yesheng Liang、Zhijian Liu，ICML 2026）。
- 文章与评测数据：[官方项目页](https://z-lab.ai/projects/dflash/)；完整实验设置和结果见[论文第 5 节](https://arxiv.org/html/2602.06036v2#S5)。这些结果属于官方文本模型实验，本仓库的 TTS 实测应以运行生成的 JSON 为准。
- 论文使用的训练数据：[NVIDIA Nemotron Post-Training Dataset V2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) 和 [CodeAlpaca](https://github.com/sahil280114/codealpaca)。论文说明其训练集使用目标模型重新生成的回答；这里引用的是原始数据来源。
- 官方权重：[DFlash 模型集合](https://huggingface.co/collections/z-lab/dflash)；示例权重及配置文件：[Qwen3-8B-DFlash-b16 文件页](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/tree/main)。
- 官方代码与文件：[z-lab/dflash](https://github.com/z-lab/dflash)。
