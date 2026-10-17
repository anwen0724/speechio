# realtime-asr

[English](README.md)

本地语音转写库。转写音频文件，或从麦克风持续输入；识别全部在本机完成。

## 功能

- 文件转写，默认引擎为 Whisper。
- 麦克风实时转写，自动切分说话段落。
- 可选 Nemotron 引擎，支持原生流式。
- 内置 Silero 检测模型，实时转写开箱即用。

## 环境要求

- 64 位 Python 3.12。
- 实时转写需要可用的麦克风。

## 安装

从当前仓库安装：

```bash
pip install .
```

发布到 PyPI 后，`pip install realtime-asr` 效果相同。

基础安装包含 Whisper 运行程序、麦克风支持、ONNX Runtime、WebRTC VAD，以及内置的 Silero 检测模型。

Linux 上如果构建时找不到 PortAudio，先安装系统库：

```bash
sudo apt install portaudio19-dev
```

### 可选：Nemotron

Nemotron 是第二个引擎，支持原生流式，默认不安装。

```bash
pip install ".[nemotron]"
asr-install-sherpa-models --root /models/sherpa-onnx --model 560ms
```

第一条命令装运行程序，第二条下载模型。变体有 `80ms`、`560ms`、`1120ms`；省略 `--model` 会装全部三个。

## 快速开始

转写文件：

```python
from realtime_asr import transcribe_audio

result = transcribe_audio("speech.wav", model="tiny", language="zh")
print(result.text)
```

- `model="tiny"` 第一次运行会下载 Whisper 模型。
- 省略 `language` 会自动判断语言。
- `result` 是 `Transcript`，`.text` 是文字。

麦克风实时输入：

```python
from realtime_asr import transcribe_stream

for transcript in transcribe_stream(model="tiny", on_partial=print):
    print(transcript.text)
```

- `on_partial` 收到会变化的预览文字。
- 循环每段说话产出一个最终 `Transcript`。

列出输入设备并指定：

```python
from realtime_asr import list_input_devices, transcribe_stream

for device in list_input_devices():
    print(device.index, device.name, "默认" if device.is_default else "")

for transcript in transcribe_stream(model="tiny", input_device=1):
    print(transcript.text)
```

## 实时转写怎么工作

默认的 Whisper 使用快照模式。它大约每 0.2 秒把已经收到的整段音频重新转写一次，所以预览会变。一段说完后，再用完整音频转写一次，得到最终文字。

Nemotron 使用原生流式。音频不断进入，文字增量产生，延迟更低。

```python
from realtime_asr import StreamTranscriber

with StreamTranscriber(
    model="/models/sherpa-onnx/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
    engine="nemotron",
    language="zh",
) as transcriber:
    for transcript in transcriber:
        print(transcript.text)
```

`StreamTranscriber` 也接受 `on_partial` 和 `input_device`。调用 `end_input()` 停止监听并交付剩余结果。

## 模型

- **Whisper** 第一次使用时从 Hugging Face 下载。用 `download_root` 改缓存目录，用 `local_files_only=True` 禁止联网。
- 已准备好的模型目录可以离线使用，目录里必须有 `model-manifest.json`，库加载前会校验。
- **Silero VAD** 随包提供，MIT 许可，不需要下载。
- **Nemotron** 权重用 `asr-install-sherpa-models` 单独安装，之后把目录作为模型路径，并设置 `engine="nemotron"`。

## API 概览

| 入口 | 用途 |
| --- | --- |
| `transcribe_audio()` | 转写文件、字节、NumPy 数组或音频块序列 |
| `transcribe_stream()` | 从麦克风持续产出最终结果 |
| `Transcriber` | 复用同一个模型转多次 |
| `StreamTranscriber` | 控制开始、预览、结束与关闭 |
| `Session` | 手动划分段落（进阶） |
| `LiveSession` | 自动划分段落（进阶） |
| `ModelConfig` | 选择引擎、模型、设备和计算类型 |
| `Transcript` | 转写结果，含 `.text` 与 `.words` |
