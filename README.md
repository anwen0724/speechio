# realtime-asr

本地语音转写库：给一段录音，或打开麦克风，把声音变成文字。识别在本机完成，不调用云端。

- 文件：WAV 等音频文件，整段转写。
- 实时：麦克风持续输入，自动判断说话起止。
- 引擎：Whisper（默认）、Nemotron（可选，原生流式）。
- 运行环境：64 位 Python 3.12。

库只负责识别核心。它不提供云端服务、多用户服务端或浏览器界面。

## 安装

从仓库安装：

```powershell
pip install .
```

发布到 PyPI 后，也可以用 `pip install realtime-asr`。

一条命令装完文件与实时所需的一切：Whisper 运行程序、麦克风支持、检测运行库，以及内置的 Silero 检测模型（约 1.2MB）。检测模型不需要再单独下载。

Linux 如果安装时提示找不到 PortAudio，先装系统库再重试：

```bash
sudo apt install portaudio19-dev
```

### 可选：Nemotron 引擎

Nemotron 是第二个引擎，默认不装。它支持原生流式，延迟比 Whisper 快照式低。

```powershell
pip install ".[nemotron]"
asr-install-sherpa-models --root D:/models/sherpa-onnx --model 560ms
```

`560ms` 是模型变体，还有 `80ms` 和 `1120ms`；不写 `--model` 会装全部三个。第一条命令装运行程序，第二条下载模型。顺序不能反。

## 第一次转写

```python
from realtime_asr import transcribe_audio

result = transcribe_audio("a.wav", model="tiny", language="zh")
print(result.text)
```

- `model="tiny"` 第一次运行会从 HuggingFace 下载模型，之后用缓存。
- 不写 `language` 就自动判断语言。
- `result` 是 `Transcript`，`.text` 是文字。

离线或想固定模型目录：

```python
result = transcribe_audio("a.wav", model="D:/models/whisper-tiny")
```

目录里要有 `model-manifest.json`，库会校验，不联网。缓存目录用 `download_root` 指定；`local_files_only=True` 禁止联网。

## 实时转写

```python
from realtime_asr import transcribe_stream

for transcript in transcribe_stream(model="tiny", language="zh", on_partial=print):
    print("最终：", transcript.text)
```

- `on_partial` 收到预览文字，预览会变。
- `for` 循环里只出现一段说完后的最终结果。
- 默认用 Whisper 快照式：大约每 0.2 秒把已录到的整段重听一遍。说完再用完整音频转一次，得到最终稿。
- 要原生增量流式，用 Nemotron。

默认使用系统默认麦克风。查看设备并指定：

```python
from realtime_asr import list_input_devices, transcribe_stream

for device in list_input_devices():
    print(device.index, device.name, "默认" if device.is_default else "")

for transcript in transcribe_stream(model="tiny", input_device=1):
    print(transcript.text)
```

## Nemotron 实时

```python
from realtime_asr import StreamTranscriber

with StreamTranscriber(
    model="D:/models/sherpa-onnx/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
    engine="nemotron",
    language="zh",
) as transcriber:
    for transcript in transcriber:
        print(transcript.text)
```

`StreamTranscriber` 也支持 `on_partial=print`、`input_device=1`，以及手动调用 `end_input()` 结束输入。

## 名词解释

| 名词 | 意思 |
| --- | --- |
| 模型 / 权重 | 参数文件，决定识别效果 |
| 运行库 / 引擎 | 真正加载并运行模型的程序 |
| Whisper | 默认识别模型，首次使用自动下载 |
| Silero | 检测模型，判断有没有人说话；随包提供 |
| Nemotron | 可选识别模型，原生流式 |
| Transcript | 一段转写结果，`.text` 是文字 |
| 一段 / turn | 一次连续说话，从开口到停顿 |
| 快照式 | 反复重听已录到的整段，预览会被改写 |
| 原生流式 | 音频不断进入，文字增量产生 |

库不训练模型，也不写运行库。库负责：采集声音、转换格式、检测人声、切分段落、调用引擎、交付结果、释放资源。

## 更多用法

| 需求 | 做法 |
| --- | --- |
| 复用模型转多个文件 | `Transcriber(model="tiny")`，再多次调用 `transcribe_file()` |
| 转写字节或音频块 | `transcribe_audio(pcm, format=AudioFormat(16000, 1, "int16"))` |
| 词时间戳 | `transcribe_audio(..., word_timestamps=True)`，读 `result.words` |
| 手动控制每段 | [continuous_session.py](examples/continuous_session.py) |
| 自动分句 | [automatic_session.py](examples/automatic_session.py) |
| 实时预览与稳定文本 | [realtime_session.py](examples/realtime_session.py) |
| 双路实时 | [dual_realtime.py](examples/dual_realtime.py) |
| 候选与纠正 | [candidates.py](examples/candidates.py) |
| 保存识别音频 | [save_audio.py](examples/save_audio.py) |
| 键盘热键听写 | [hotkey_dictation.py](examples/hotkey_dictation.py) |
| 自定义引擎 | [custom_engine.py](examples/custom_engine.py) |

## 本地命令行

命令行只存在于仓库里，不随 wheel 发布：

```powershell
python -m apps.cli --help
python -m apps.cli file a.wav --model .runtime/models/whisper-tiny
python -m apps.cli mic --model .runtime/models/whisper-tiny
```

`config`、`models`、`devices`、`mic`、`file`、`stdin` 六个子命令，用 `--help` 查看参数。

## 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 首次运行卡住 | 正在下载 Whisper 模型；检查网络，或改用本地模型目录 |
| 找不到麦克风 | 检查系统权限，或用 `list_input_devices()` 选设备 |
| Linux 安装失败 | 安装 `portaudio19-dev` 后重试 |
| 用 Nemotron 报缺少运行库 | 安装 `.[nemotron]` |
| Python 3.13 | 不支持，使用 3.12 |

## 开发

```powershell
uv sync --locked --extra nemotron
uv run --no-sync python -m unittest discover -s tests -t .
```

目录：

- `src/realtime_asr/`：库。
- `apps/cli/`：本地命令行，不发布。
- `examples/`：可运行示例。
- `evals/`：评估入口。

内置检测模型来自 Silero，MIT 许可，说明见 [assets/README.md](src/realtime_asr/assets/README.md)。
