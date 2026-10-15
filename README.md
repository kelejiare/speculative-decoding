# MiniCPM-o 4.5 语音推理优化：DFlash 扩展实验

本仓库是个人项目「MiniCPM-o 4.5 语音推理优化与投机解码探索」中 DFlash 扩展实验的公开代码，基于开源实现开展 TTS 推理复现与分析，提供文本/图片/视频转语音、录制视频双工推理及接受率评测入口。

## 项目背景与公开范围

整体项目围绕 MiniCPM-o 4.5 的 Thinker、Talker 与 Code2Wav 链路开展语音推理优化，包含两个方向：

| 方向 | 环境与工作内容 | 公开范围 |
| --- | --- | --- |
| 昇腾推理优化 | OpenBMB × 华为挑战赛；基于 vLLM-Omni 与 Ascend 910C 开展部署、流式链路分析、Code2Wav 优化与 Talker 请求批处理 | 作为项目背景，相关实现与评测材料未包含在本仓库 |
| DFlash 扩展实验 | 个人探索；基于 PyTorch/Transformers 与 CUDA，在 TTS 阶段运行草稿模型与 Talker 协同解码 | 本仓库提供该部分推理代码、评测入口与使用文档 |

昇腾侧吞吐、TTFP、E2E 和 RTF 优化结果对应其独立实验设置，不能作为本仓库的 DFlash 性能结论。目前仓库未提交 DFlash 与原生解码的完整对照结果；评测脚本可用于生成接受率等实验记录。

DFlash 方法参考[官方论文](https://arxiv.org/abs/2602.06036)：用轻量级块扩散模型并行生成草稿，再由目标模型验证。

## MiniCPM-o 与 DFlash 的分工

| 部分 | 在本实验中的作用 | 对应资源 |
| --- | --- | --- |
| MiniCPM-o 4.5 | 提供多模态理解、文本生成、Talker/TTS 和语音波形生成能力 | [官方模型说明](https://huggingface.co/openbmb/MiniCPM-o-4_5)、[基础权重及配置文件](https://huggingface.co/openbmb/MiniCPM-o-4_5/tree/main) |
| DFlash | 提供块扩散草稿与投机验证方法 | [官方论文](https://arxiv.org/abs/2602.06036)、[官方代码](https://github.com/z-lab/dflash) |
| 本仓库实验 | 在 MiniCPM-o 的 TTS codec token 解码阶段使用兼容的 DFlash 草稿模型，运行推理并记录接受率等指标 | 本地 TTS 草稿权重、`outputs/` 中的运行结果 |

结合流程：输入 → MiniCPM-o 多模态理解与文本生成 → DFlash 提议语音 token、MiniCPM-o Talker 验证 → Token2wav 生成波形。DFlash 在此用于 TTS 解码，主模型文本生成及视觉、音频编码保持原有流程。

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

# 录制视频双工推理：请自行提供输入视频
uv run minicpmo-dflash-duplex --draft-model /path/to/tts-draft --video /path/to/input.mp4

# DFlash 接受率评测
uv run minicpmo-dflash-benchmark --draft-model /path/to/tts-draft --output outputs/benchmark.json
```

常用参数：`--ref-audio` 指定参考音色，`--device cuda:0` 选择 GPU，`--vocoder-device cuda:1` 将声码器放到另一张显卡。Simplex/Duplex 可通过 `--disable-dflash` 切换为原生 TTS 解码；接受率评测命令不支持此选项。各命令的完整参数可通过 `--help` 查看。

生成文件默认保存在 `outputs/`。评测应区分 TTS 解码时间与完整推理耗时；接受率也不等同于端到端加速比。

## MiniCPM-o 官方参考

- 模型介绍、使用示例与评测数据：[MiniCPM-o 4.5 模型卡](https://huggingface.co/openbmb/MiniCPM-o-4_5)。
- 基础模型权重、配置及推理文件：[官方文件目录](https://huggingface.co/openbmb/MiniCPM-o-4_5/tree/main)。
- 官方项目代码：[OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)（MiniCPM-o 官方仓库链接当前指向此处）。

## DFlash 官方参考

- 论文：[DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)（Jian Chen、Yesheng Liang、Zhijian Liu，ICML 2026）。
- 文章与评测数据：[官方项目页](https://z-lab.ai/projects/dflash/)；完整实验设置和结果见[论文第 5 节](https://arxiv.org/html/2602.06036v2#S5)。这些结果属于官方文本模型实验，本仓库的 TTS 实测应以运行生成的 JSON 为准。
- 论文使用的训练数据：[NVIDIA Nemotron Post-Training Dataset V2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) 和 [CodeAlpaca](https://github.com/sahil280114/codealpaca)。论文说明其训练集使用目标模型重新生成的回答；这里引用的是原始数据来源。
- 官方权重：[DFlash 模型集合](https://huggingface.co/collections/z-lab/dflash)；示例权重及配置文件：[Qwen3-8B-DFlash-b16 文件页](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/tree/main)。
- 官方代码与文件：[z-lab/dflash](https://github.com/z-lab/dflash)。

以上 MiniCPM-o 评测、DFlash 论文数据和本仓库 TTS 实测属于不同实验设置，应分别引用。DFlash 论文使用的数据集不代表本实验的 TTS 草稿训练数据。
