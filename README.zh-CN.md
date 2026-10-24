# speechio

[English](https://github.com/anwen0724/speechio/blob/main/README.md)

speechio 是一个 Python 语音处理库，提供语音识别、语音合成和声音克隆。你可以使用本地模型，也可以通过 API Key 调用已接入的云端服务，将语音能力加入自己的应用。

## 主要功能

- **语音识别**：将音频文件、内存音频或麦克风声音转换为文字，支持实时结果与分段识别。
- **语音合成**：将文字转换为完整音频，或使用支持流式输出的模型逐块接收音频。
- **声音克隆**：根据参考录音创建音色，保存后重复使用，也可以提供参考录音直接合成。
- **模型扩展**：通过自定义适配器接入其他本地模型或云端服务。
- **示例应用**：在浏览器中体验识别、合成和克隆，通过调试台查看参数与事件。

具体能力取决于所选模型。对话回复、业务流程和音频播放由使用 speechio 的应用实现。

## 安装

需要 Python 3.11–3.13：

```bash
pip install speechio
```

默认安装包含云端调用、音频处理、麦克风、语音活动检测和唤醒词所需的依赖。云端调用需要你自己的服务凭据；部分唤醒词引擎还需要模型文件或 AccessKey。

使用本地识别或合成模型时，选择对应的可选依赖组：

| 本地模型 | 安装命令 |
| --- | --- |
| Whisper | `pip install "speechio[stt-whisper]"` |
| Nemotron | `pip install "speechio[stt-nemotron]"` |
| Qwen3-TTS Base | `pip install "speechio[tts-qwen]"` |
| Kokoro | `pip install "speechio[tts-kokoro]"` |

安装依赖不会自动下载识别或合成模型。已有完整模型目录时可直接加载；否则先单独下载。系统依赖、Python 版本和 GPU 支持因运行库而异，见[安装与运行环境](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#2-安装与运行环境)。

## 快速开始

详细使用介绍请阅读[基本使用指南](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md)，其中包含模型配置、接口参数、语音识别、语音合成和声音克隆的使用方法。

以下两个示例使用阿里云百炼：

```python
import os

from speechio import RemoteModelConfig, TranscriptionOptions, transcribe_audio

config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-asr-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
result = transcribe_audio(
    "speech.wav",
    config=config,
    options=TranscriptionOptions(language="zh"),
)
print(result.text)
```

该短音频型号还检查请求大小。较长录音可选择文件任务型号。麦克风、实时音频和长文件的用法见[语音识别指南](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#4-将音频转换为文字)。

### 文字转音频

指定模型支持的音色，生成并保存 WAV 文件：

```python
import os

from speechio import RemoteModelConfig, SynthesisOptions, synthesize

config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-tts-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
audio = synthesize(
    "你好，欢迎开始今天的练习。",
    config=config,
    options=SynthesisOptions(voice="yuxiaoyun_v3.1", language="zh"),
)
audio.save("hello.wav")
```

`hello.wav` 是生成的音频，默认不会覆盖已有文件。流式输出见[语音合成指南](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#5-将文字转换为音频)。使用自己的参考录音创建音色，见[声音克隆与复用](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#6-克隆声音与复用音色)。

使用本地模型时，通过 `LocalModelConfig(adapter=..., model_path=...)` 指定适配器与已有模型目录。下载和配置步骤见[模型配置指南](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md#3-选择与配置模型)。

## 支持的模型

| 模型 | 运行位置 | 主要能力 |
| --- | --- | --- |
| Whisper | 本地 | 文件识别；通过重复识别已收到的音频更新实时文字 |
| Nemotron 3.5 ASR 0.6B | 本地 | 文件识别、原生流式识别；当前适配器只支持 CPU |
| Qwen3-TTS 12Hz Base，0.6B / 1.7B | 本地 | 完整语音合成、声音克隆；当前适配器不提供流式合成 |
| Kokoro 82M v1.1-zh | 本地 | 中英文合成、按文本片段输出音频；不支持声音克隆 |
| `qwen-audio-3.1-asr-flash` | 阿里云百炼 | 短音频整段识别 |
| `qwen-audio-3.1-asr-flash-streaming` | 阿里云百炼 | 原生流式识别，也可处理完整音频输入 |
| `qwen-audio-3.1-asr-flash-filetrans` | 阿里云百炼 | 长音频文件识别，支持本地文件和服务可访问的 URL |
| `qwen-audio-3.1-tts-flash` | 阿里云百炼 | 完整与流式合成、持续追加文本、声音克隆 |

本地调用使用已有模型文件；云端调用将输入发送给所选服务。输入限制、设备要求和音色兼容性见[基本使用指南](https://github.com/anwen0724/speechio/blob/main/docs/basic-usage.md)。

## Web 示例应用

仓库提供示例应用与调试台，支持录音和文件识别、文字合成、声音克隆与音色复用。调试台还提供参数和事件查看。两者调用同一套核心包接口，不随 `pip install speechio` 安装。

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，克隆仓库并进入目录：

```bash
git clone https://github.com/anwen0724/speechio.git
cd speechio
```

首次使用时，将 `apps/web/config.example.toml` 复制为 `apps/web/config.toml`。已有配置时，直接编辑该文件：

- 使用云端模型：填写对应条目的 `apikey`，或将 `apps/web/.env.example` 复制为 `apps/web/.env` 并填写密钥。读取优先级为配置中的 `apikey`、进程环境变量、`.env`，空白值跳过。
- 使用本地模型：填写已有模型目录，并启用配置示例中的对应条目。

使用云端模型时，启动命令为：

```bash
uv run --group web python -m apps.web --enable-debug
```

使用本地模型时，在启动命令中添加对应的 `--extra`，例如 Whisper：

```bash
uv run --group web --extra stt-whisper python -m apps.web --enable-debug
```

其他本地模型对应 `stt-nemotron`、`tts-qwen`、`tts-kokoro`。使用多组时重复添加 `--extra`。这些参数安装运行依赖，不下载模型，也不保证 GPU 运行环境已就绪。

- 示例应用：<http://127.0.0.1:8767/>
- 调试台：<http://127.0.0.1:8767/debug>

在启动终端按 `Ctrl+C` 停止服务。不需要调试台时省略 `--enable-debug`。应用仅监听本机地址，同一时间运行一个任务；本机配置和密钥文件已加入 Git 忽略规则。

## 许可证

speechio 使用 [MIT 许可证](https://github.com/anwen0724/speechio/blob/main/LICENSE)，Copyright © 2026 anwen。

内置 Silero 检测模型的许可随包提供，见[模型资源说明](https://github.com/anwen0724/speechio/blob/main/src/speechio/assets/README.md)。其他模型、依赖和云端服务遵循各自的许可与使用条款。
