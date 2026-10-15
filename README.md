# MiniCPM-o 4.5 语音推理与 DFlash 投机解码

本仓库用于复现和研究 MiniCPM-o 4.5 的语音投机解码。基于开源实现，将 DFlash 草稿模型用于语音合成阶段，提供文本、图片和视频输入的语音生成，以及草稿接受率评测。

## 项目背景

这项实验是个人「MiniCPM-o 4.5 语音推理优化」项目的延伸。此前在 [MiniCPM & 昇腾推理优化与应用创新挑战赛](https://ascend.openbmb.cn/competition)中，基于 vLLM-Omni 和 Ascend 910C 开展模型部署、流式音频处理、声码器优化及 Talker 请求批处理。

本仓库公开的是后续基于 PyTorch、Transformers 和 CUDA 开展的 DFlash 实验代码。昇腾侧代码与评测材料未收录；此前的性能提升也不作为 DFlash 的实验结果。目前仓库尚未提供 DFlash 与原生解码的完整性能对照。

## 工作原理

MiniCPM-o 负责理解输入并生成文本和语音。DFlash 在其中的语音解码阶段，一次预测多个候选语音 token，再交给 MiniCPM-o 的 Talker 验证；通过验证的候选被保留，未通过时按采样规则重新选取。最后由 Token2wav 将语音 token 转为可播放的音频。

```text
文本 / 图片 / 视频
        ↓
MiniCPM-o 理解输入并生成文本
        ↓
DFlash 生成候选语音 token → Talker 验证
        ↓
Token2wav 合成音频
```

DFlash 的方法见[官方论文](https://arxiv.org/abs/2602.06036)。本实验将它用于 TTS 解码，模型的文本生成、图像编码和音频编码仍由 MiniCPM-o 完成。

## 环境与模型准备

需要 Python 3.10、NVIDIA CUDA 显卡、FFmpeg 和 uv。安装好这些工具后，执行：

```bash
git clone https://github.com/kelejiare/speculative-decoding.git
cd speculative-decoding
uv sync
```

基础模型默认使用 `openbmb/MiniCPM-o-4_5`，首次运行会下载，也可以通过 `--model` 指定本地目录。

以下命令中的 `/path/to/tts-draft` 是占位路径，请替换成兼容 MiniCPM-o Talker 的 DFlash 草稿权重目录。目录需要包含 `config.json` 和 `model.safetensors`；配置、权重结构及权重中的词表映射须符合本项目加载器的要求。当前加载器要求 `speculators_model_type` 为 `dflash`，且 `sample_from_anchor` 为 `false`。

DFlash 官方发布的 Qwen 等文本模型草稿权重用于对应的目标模型，不能直接替代这里的 TTS 草稿权重。

## 运行示例

```bash
# 文本转语音
uv run minicpmo-dflash-simplex \
  --draft-model /path/to/tts-draft \
  --text "你好，这是一段语音合成测试。" \
  --output outputs/demo.wav

# 按秒读取视频和音频，运行双工推理
uv run minicpmo-dflash-duplex \
  --draft-model /path/to/tts-draft \
  --video /path/to/input.mp4

# 运行内置样例，统计草稿接受率并保存结果
uv run minicpmo-dflash-benchmark \
  --draft-model /path/to/tts-draft \
  --output outputs/benchmark.json
```

双工示例读取本地视频文件，需自行准备输入。它模拟连续音视频输入的处理过程；运行这些命令不会启动摄像头或麦克风。

常用参数：

| 参数 | 用途 |
| --- | --- |
| `--ref-audio` | 指定参考音色的音频文件 |
| `--device cuda:0` | 选择运行主模型和草稿模型的显卡 |
| `--vocoder-device cuda:1` | 将声码器放到另一张显卡，减轻主显卡的显存压力 |
| `--disable-dflash` | 在单轮或双工推理中使用原生 TTS 解码，评测命令不支持此选项 |
| `--help` | 查看对应命令的完整参数 |

## 如何看评测结果

生成的音频、视频和 JSON 默认保存在 `outputs/`，该目录不纳入 Git 版本管理。评测脚本内置 20 条中英文样例，涵盖朗读、问答、翻译等任务，可统计草稿接受率、每轮接受的草稿 token 数、解码耗时和整体运行耗时。

接受率高不一定意味着运行更快。判断实际收益，需要在相同硬件、输入和采样配置下比较原生解码与 DFlash，并结合生成音频检查质量。当前评测命令只运行 DFlash，不会自动完成两者的对照。

## 官方资料

**MiniCPM-o 4.5**

- [模型介绍、使用示例与评测结果](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- [基础模型权重、配置与推理文件](https://huggingface.co/openbmb/MiniCPM-o-4_5/tree/main)
- [OpenBMB 官方代码](https://github.com/OpenBMB/MiniCPM-V)

**DFlash**

- [论文：DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)（Jian Chen、Yesheng Liang、Zhijian Liu，ICML 2026）
- [官方文章](https://z-lab.ai/projects/dflash/)与[论文实验设置和结果](https://arxiv.org/html/2602.06036v2#S5)
- [官方代码](https://github.com/z-lab/dflash)与[模型权重集合](https://huggingface.co/collections/z-lab/dflash)
- [Qwen3-8B 草稿模型的权重及配置文件](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/tree/main)（文本模型示例）
- 论文使用的原始训练数据：[NVIDIA Nemotron Post-Training Dataset V2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)、[CodeAlpaca](https://github.com/sahil280114/codealpaca)。论文使用目标模型重新生成回答来构建训练集。

上述官方评测和数据集用于了解各自模型与方法，不代表本仓库的 TTS 实测结果或草稿模型训练数据。
