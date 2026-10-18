# speechio

[English](https://github.com/anwen0724/speechio/blob/main/README.md)

SpeechIO 是一个 Python 语音转写库。你可以把音频文件转成文字，也可以接入麦克风，边说边显示文字，并在每段说话结束后取得最终结果。

支持本地模型和云端服务：本地识别使用已有模型目录，音频在本机处理；云端识别使用你的 API Key，将音频发送给所选服务。

## 功能

- 转写音频文件、PCM 音频和音频块序列。
- 从麦克风持续采集，自动检测语音和停顿，按段返回结果。
- 在说话过程中提供实时文字回调。
- 支持手动分段、唤醒词和自定义识别引擎。
- 提供 Web 示例应用和调试台，便于体验和检查调用结果。

当前已适配：

| 模型 | 运行位置 | 文件转写 | 说话过程中的文字 |
| --- | --- | --- | --- |
| Whisper | 本地 | 支持 | 定期识别已收到的音频，更新预览 |
| Nemotron 3.5 ASR 0.6B | 本地 | 支持 | 原生流式返回 |
| `qwen-audio-3.1-asr-flash` | 阿里云百炼 | 支持短音频 | 不支持；可在分段结束后识别 |
| `qwen-audio-3.1-asr-flash-streaming` | 阿里云百炼 | 支持短音频 | 原生流式返回 |

百炼首版提供短音频和实时识别，不包含长录音文件任务。整段调用最多接受 300 秒音频，HTTP 调用还受编码后大小限制，超限会报错。

## 安装

需要 64 位 Python 3.11–3.13。

```bash
python -m pip install speechio
```

按所需功能安装可选依赖，以下命令也会安装 SpeechIO：

| 用途 | 命令 |
| --- | --- |
| 麦克风输入 | `python -m pip install "speechio[microphone]"` |
| 百炼云端识别 | `python -m pip install "speechio[remote]"` |
| 百炼 + 麦克风 | `python -m pip install "speechio[remote,microphone]"` |
| 本地 Nemotron | `python -m pip install "speechio[nemotron]"` |
| OpenWakeWord 唤醒词 | `python -m pip install "speechio[wake-openwakeword]"` |
| Porcupine 唤醒词 | `python -m pip install "speechio[wake-porcupine]"` |

这些命令安装库和依赖，不会下载语音识别模型。使用本地识别时，还需按下面的示例准备模型目录。

在 Linux 或 macOS 上安装麦克风依赖前，还需要准备 PortAudio，具体见 [PyAudio 安装说明](https://pypi.org/project/PyAudio/)。唤醒词还需要相应模型文件；Porcupine 需要 AccessKey。

## 快速开始

下面用 Whisper base 在 CPU 上转写一个本地音频文件。

先单独运行一次下载代码，把模型保存到指定目录。已有可供 faster-whisper 使用的模型目录时，可以跳过下载。

```python
from speechio.models import download_model

download_model(
    engine="whisper",
    model_id="base",
    destination="models/whisper-base",
)
```

将待识别的音频保存为 `speech.wav`，然后运行：

```python
from speechio import LocalModelConfig, transcribe_audio

config = LocalModelConfig(
    engine="whisper",
    model_path="models/whisper-base",
    device="cpu",
    compute_type="int8",
)

result = transcribe_audio("speech.wav", config=config, language="zh")
print(result.text)
```

以上相对路径都以运行 Python 时的工作目录为准。识别时只加载 `model_path` 中已有的文件，不会自动下载模型。`language="zh"` 指定中文，英文使用 `"en"`，省略则不指定语言提示。

## 使用示例

### 云端转写

安装 `remote` 可选依赖，并将百炼密钥设置为 `DASHSCOPE_API_KEY` 环境变量。以下代码读取该变量；SDK 本身不会自动读取 `.env` 文件。

```python
import os

from speechio import RemoteModelConfig, transcribe_audio

config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-asr-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)

result = transcribe_audio("speech.wav", config=config, language="zh")
print(result.text)
```

也可以通过 `api_key` 直接传入你的应用取得的密钥。调用会将音频发送至百炼，并按你的账户和所选模型计费。

### 麦克风实时转写

安装 `microphone` 可选依赖，并准备好快速开始中的模型目录。运行后对着默认麦克风说话，按 `Ctrl+C` 退出。

```python
from speechio import LocalModelConfig, StreamTranscriber

config = LocalModelConfig(
    engine="whisper",
    model_path="models/whisper-base",
    device="cpu",
    compute_type="int8",
)

try:
    with StreamTranscriber(
        config=config,
        language="zh",
        on_partial=lambda text: print("实时：", text),
    ) as stream:
        for result in stream:
            print("最终：", result.text)
except KeyboardInterrupt:
    pass
```

`on_partial` 收到的文字可能变化，循环取得的是每段最终结果。语音和停顿由随包提供的 Silero 检测模型判断，不需要额外下载检测模型。需要选麦克风时，可用 `list_input_devices()` 查看设备编号，再传入 `input_device`。

Whisper 在说话时更新预览，在每段结束后完整识别该段音频。Nemotron 和百炼流式模型默认直接采用流式任务的最终结果；设置 `completion_mode="final"` 才会额外识别整段音频。云端重识别会增加请求，可能增加费用。

使用云端麦克风转写时，安装 `remote,microphone`，把上例的 `config` 换成 `RemoteModelConfig`，模型选择 `qwen-audio-3.1-asr-flash-streaming`。需要停止采集后继续获取尾部结果时，调用 `stream.end_input()` 并继续消费循环。

更多场景可参考仓库中的 [示例代码](https://github.com/anwen0724/speechio/tree/main/examples)。

## Web 示例应用

仓库提供两个页面：示例应用用于录音和文件转写，调试台用于调整参数、查看事件和调用示例。两者使用同一套核心包 API。通过 pip 安装的 SDK 不包含 Web 应用。

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再克隆仓库。已有仓库时，直接进入仓库根目录：

```bash
git clone https://github.com/anwen0724/speechio.git
cd speechio
```

首次使用时，将 `apps/web/config.example.toml` 复制为 `apps/web/config.toml`，已有配置则直接编辑。

在配置中填写本地模型目录，或填写云端模型的 `apikey`。也可以将 `apps/web/.env.example` 复制为 `apps/web/.env` 并填入密钥。Web 的读取优先级是：配置文件中的 `apikey`、进程环境变量、`apps/web/.env`；空白值跳过。

```bash
uv run --group web python -m apps.web --enable-debug
```

- 示例应用：<http://127.0.0.1:8767/>
- 调试台：<http://127.0.0.1:8767/debug>

在启动终端按 `Ctrl+C` 关闭服务。不需要调试台时，省略 `--enable-debug`。应用仅监听本机地址，同一时间运行一个任务；本机配置和密钥文件已加入 Git 忽略规则。

## 许可证

SpeechIO 使用 [MIT 许可证](https://github.com/anwen0724/speechio/blob/main/LICENSE)，Copyright © 2026 anwen。

内置 Silero 模型的许可说明随包提供，见 [模型资源说明](https://github.com/anwen0724/speechio/blob/main/src/speechio/assets/README.md)。其他模型、依赖和云端服务分别遵循其提供方的许可与使用条款。
