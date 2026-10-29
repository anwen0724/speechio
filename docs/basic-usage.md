# speechio 基本使用

speechio 是一个 Python 语音处理库。你可以把它导入自己的程序，用本地模型或云端服务将音频转换为文字、将文字转换为音频。你也可以克隆声音，并重复使用克隆得到的音色。

本文描述当前仓库的公开接口。示例面向在其他项目中安装和使用 speechio 的开发者，不需要运行 speechio 的 Web 示例应用。本文新增的能力说明字段和 `list_voices()` 尚未发布到 PyPI，使用已发布版本时暂不可调用这些新接口。

## 1. speechio 提供什么

| 能力 | 输入 | 输出 |
| --- | --- | --- |
| 语音识别（STT） | 音频文件、音频数据或麦克风声音 | 文字；模型支持时也可返回词时间信息 |
| 语音合成（TTS） | 文字、音色和合成设置 | 完整音频或连续返回的音频块 |
| 声音克隆 | 参考录音和指定模型 | 可保存、可再次用于合成的 `Voice` |
| 模型接入 | 本地模型配置、云端服务配置或自定义适配器 | 通过相同的使用入口调用所选模型 |

speechio 负责音频处理、模型调用及任务资源管理。对话回复、提示词、聊天记录、评分、用户管理、页面和音频播放由应用实现。例如，对话应用可以用 speechio 识别用户说话，将文字交给自己的对话模型，再用 speechio 合成回复。

按任务选择入口：

| 需要完成的任务 | 使用入口 |
| --- | --- |
| 识别一个文件或一段音频 | `transcribe_audio()` |
| 连续识别多个文件，复用同一个模型 | `Transcriber` |
| 采集本机麦克风并持续识别 | `transcribe_microphone()` 或 `MicrophoneTranscriber` |
| 接收应用传入的连续音频，控制分段和读取事件 | `STTRuntime` 与会话对象 |
| 合成一次语音 | `synthesize()` |
| 多次合成或流式合成 | `Synthesizer` |
| 独立克隆声音 | `clone_voice()` 或 `Synthesizer.clone_voice()` |
| 查询已选模型的能力 | 处理对象的 `capabilities` 属性 |
| 查询已选合成模型的可用音色 | `Synthesizer.list_voices()` |

## 2. 安装与运行环境

在你的项目使用的 Python 环境中安装：

```powershell
pip install speechio
```

默认安装包含音频读写、格式转换、云端调用、麦克风采集、语音活动检测、唤醒词和模型下载所需的依赖。调用已接入的云端服务不需要另外安装 `remote` 组；使用麦克风也不需要另外安装 `microphone` 组。

使用本地识别或合成模型时，按模型安装可选依赖：

| 本地模型 | 安装命令 | 增加的主要运行库 |
| --- | --- | --- |
| Whisper | `pip install "speechio[stt-whisper]"` | faster-whisper，以 CTranslate2 执行 Whisper |
| Nemotron | `pip install "speechio[stt-nemotron]"` | sherpa-onnx，执行已适配的 ONNX 模型 |
| Qwen3-TTS Base | `pip install "speechio[tts-qwen]"` | qwen-tts、PyTorch 和 torchaudio |
| Kokoro | `pip install "speechio[tts-kokoro]"` | Kokoro、Misaki 中英文文本处理和 PyTorch |

可以在一条命令中选择多组：

```powershell
pip install "speechio[stt-whisper,tts-kokoro]"
```

这些依赖安装在同一个 Python 环境中，不会各自创建环境。pip 会解析共同依赖；如果所选版本无法共存，应更换兼容版本或分别部署在不同环境中。

安装不会下载 Whisper、Qwen 等识别或合成模型文件。随包提供的语音活动检测模型只用于判断音频中是否有人说话。

### Python、系统和设备

- 主包支持 Python 3.11–3.13。要使用本文的全部本地模型示例，可从 Python 3.12 开始。Kokoro 0.9.4 和 Misaki 0.9.4 要求 Python 低于 3.13。
- Windows 已有本地模型运行验证。macOS 是否可运行所选模型，还取决于对应运行库；本文不把未实测的组合列为已验证支持。
- macOS 安装 PyAudio 前通常需要 `brew install portaudio`；从源码构建时还需要 Xcode 命令行工具。参见 [PyAudio 安装说明](https://people.csail.mit.edu/hubert/pyaudio/)。
- 接口使用 `device="cpu"` 或 `device="gpu"`。当前 GPU 接入使用 NVIDIA CUDA，不支持 Apple MPS。
- Qwen 和 Kokoro 的 GPU 模式需要可用的 CUDA 版 PyTorch。安装可选组不代表显卡驱动和 GPU 版运行库已经就绪。Whisper 使用 CTranslate2，不使用 PyTorch。
- 当前 Nemotron INT8 适配器只支持 CPU。

## 3. 选择与配置模型

先确定任务，再选择支持该任务的模型。统一接口不代表每个模型都支持文件、实时识别、流式合成和声音克隆。

### 本地模型配置

`LocalModelConfig` 指定适配器和已经存在的模型目录：

```python
from speechio import LocalModelConfig

asr_config = LocalModelConfig(
    adapter="whisper",
    model_path="models/whisper-base",
    device="cpu",
)
```

示例中的相对路径以程序的当前工作目录为起点，也可以填写绝对路径。

| 字段 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `adapter` | 是 | 无 | `whisper`、`nemotron`、`qwen_tts` 或 `kokoro` |
| `model_path` | 是 | 无 | 该适配器能读取的完整模型目录 |
| `device` | 否 | `"cpu"` | 执行设备；可选 `"cpu"`、`"gpu"`，受模型支持范围限制 |
| `device_index` | 否 | `0` | GPU 序号；不是麦克风序号 |
| `compute_type` | 否 | `None` | 计算精度；省略时由适配器选择 |
| `adapter_options` | 否 | `None` | 适配器专用设置；普通调用可以省略 |

模型目录中的文件格式应与适配器匹配。例如，Whisper 使用 faster-whisper 可读取的 CTranslate2 模型，不能直接填写任意 Whisper 权重文件的路径。

精度也不是所有模型通用的选项。Qwen 默认在 CPU 使用 `float32`、在 GPU 使用 `bfloat16`；Kokoro 使用 `float32`；当前 Nemotron 使用 INT8。不要把某个模型的精度设置直接复制给另一个模型。

### 下载与加载分开

**已有完整模型目录时，直接使用 `LocalModelConfig`，无需调用下载方法。** 加载过程中缺少文件会报错，不会自动下载。

还没有模型时，可以单独运行：

```python
from speechio.models import download_model

model_path = download_model(
    engine="whisper",
    model_id="base",
    destination="models/whisper-base",
)
print(model_path)
```

三个参数都需要提供。`engine` 是下载接口中表示适配器的字段；`model_id` 选择要下载的模型；`destination` 是本地存放目录。返回值是模型目录的绝对路径，可直接传给 `LocalModelConfig.model_path`。

| `engine` | `model_id` 示例 |
| --- | --- |
| `whisper` | `base`、`large-v3`、`large-v3-turbo` |
| `nemotron` | `560ms` |
| `qwen_tts` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`、`Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `kokoro` | `hexgrad/Kokoro-82M-v1.1-zh` |

Kokoro 的下载包括预设音色和英文文本处理资源。下载完成后仍需安装对应可选依赖，才能执行模型。

### 云端模型配置

`RemoteModelConfig` 指定服务商、模型及访问凭据。以下示例读取应用进程中的环境变量：

```python
import os
from speechio import RemoteModelConfig

asr_config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-asr-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
```

运行前应由应用设置 `DASHSCOPE_API_KEY`。也可以由应用读取配置文件后，把密钥字符串传给 `api_key`。speechio 不读取 TOML 或 `.env` 文件；仅创建 `.env` 文件不会让其中的内容自动进入环境变量。

| 字段 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `provider` | 是 | 无 | 已接入的云端服务商；当前为 `"bailian"` |
| `model` | 是 | 无 | 已接入的模型名称，见下表 |
| `api_key` | 否，但百炼调用需要 | `None` | 应用明确传入的 API 密钥 |
| `base_url` | 否 | `None` | 省略时使用默认服务地址；百炼为 `https://dashscope.aliyuncs.com` |
| `timeout_seconds` | 否 | `30.0` | 网络操作的等待设置，不是音频时长上限 |

以下是 0.2.0 已接入的云端型号及其主要用途。填写其他型号不会自动获得对应适配能力。

| `model` | 用途与边界 |
| --- | --- |
| `qwen-audio-3.1-asr-flash` | 短音频整段识别；适配器限制为 300 秒，并检查请求大小 |
| `qwen-audio-3.1-asr-flash-streaming` | 原生实时识别，也可识别完整输入；speechio 不再施加 300 秒限制，仍受服务规则约束 |
| `qwen-audio-3.1-asr-flash-filetrans` | 长音频文件任务，支持本地文件和服务可访问的 URL；不接收麦克风或裸音频块 |
| `qwen-audio-3.1-tts-flash` | 完整合成、音频流、追加文本合成和声音克隆 |

## 4. 将音频转换为文字

### 识别一个文件

下面使用云端短音频模型。准备一份不超过模型限制的 `meeting.m4a` 和环境变量中的密钥：

```python
import os
from speechio import RemoteModelConfig, TranscriptionOptions, transcribe_audio

asr_config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-asr-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
result = transcribe_audio(
    "meeting.m4a",
    config=asr_config,
    options=TranscriptionOptions(language="zh"),
)
print(result.text)
```

换成本地模型时，将 `asr_config` 替换为第 3 节的 `LocalModelConfig`。本地文件支持 WAV、MP3、M4A、FLAC、OGG 等常见格式，文件内应有可解码的音轨；直接上传到云端文件任务时，还应符合服务支持的格式。

`transcribe_audio()` 的 `audio` 和 `config` 必填。文件路径不需要提供 `format`；裸采样数据需要。`options=None` 使用默认识别选项，`max_audio_seconds=None` 表示 speechio 不额外限制整段输入时长。可把后者设为正数秒数，限制应用接受的音频长度。它不会取消云端模型自己的限制，也不意味着内存用量不受音频长度影响。

`TranscriptionOptions` 两个字段均可省略：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `language` | `None` | 不指定语言；可用 `"zh"`、`"en"` 等所选模型支持的语言 |
| `word_timestamps` | `False` | 是否请求词时间信息；模型不支持时不会补造 |

返回的 `Transcript` 中，`text` 是识别文字，`requested_language` 是请求语言，`reported_language` 是模型报告的语言，后者可能为空。请求时间信息后，检查 `timestamp_status`：`reported` 表示提供了时间信息，`unavailable` 表示无法提供；未请求时为 `not_requested`。

`words` 中每项的 `start_sample`、`end_sample` 使用标准化后的 16 kHz 采样位置，除以 `16000` 可得到秒数。

### 多次识别与内存音频

顶层函数适合单次调用。重复调用本地模型时，使用 `Transcriber` 避免每次重新加载：

```python
from speechio import LocalModelConfig, Transcriber

asr_config = LocalModelConfig(adapter="whisper", model_path="models/whisper-base")
with Transcriber(asr_config) as transcriber:
    print(transcriber.transcribe_file("first.wav").text)
    print(transcriber.transcribe_file("second.m4a").text)
```

应用已有录音数据时，可以直接传入裸采样数据。以下函数的 `pcm_bytes` 由调用方提供，内容为 16 kHz、单声道、小端 PCM16，不是完整 WAV 文件的字节：

```python
from speechio import AudioFormat, Transcriber

def recognize_pcm(pcm_bytes, asr_config):
    with Transcriber(asr_config) as transcriber:
        return transcriber.transcribe(
            pcm_bytes,
            format=AudioFormat(sample_rate=16000, channels=1, sample_type="int16"),
        )
```

`AudioFormat` 的上述三个值也是默认值。除了字节，识别入口也接收格式匹配的 NumPy 数组；`float32` 采样范围为 `[-1, 1]`，多声道数组形状为 `(帧数, 声道数)`。

`transcribe_chunks(chunks, *, format, options=None)` 接收连续音频块的迭代器。它等全部输入结束后返回一个 `Transcript`，不是实时文字迭代器。

### 识别长文件或 URL

长录音可以选择文件任务模型：

```python
import os
from speechio import RemoteModelConfig, Transcriber

asr_config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-asr-flash-filetrans",
    api_key=os.environ["DASHSCOPE_API_KEY"],
)
with Transcriber(asr_config) as transcriber:
    result = transcriber.transcribe_file("long-meeting.m4a", task_timeout_seconds=3600)
    print(result.text)
```

该适配器对本地上传检查 1 GiB 文件大小和 12 小时时长上限。`task_timeout_seconds` 默认 `3600.0`，控制文件任务的总等待时间，与单次网络操作的 `timeout_seconds` 分开设置。

如果文件已有可被服务读取的 URL，可以改用 `transcriber.transcribe_url(url)`，或单次入口 `transcribe_url(url, config=asr_config)`。URL 不是 `file:///` 本机路径；服务端需要能够访问它。普通本地模型没有这个 URL 入口的能力。

### 使用麦克风

`transcribe_microphone()` 采集**运行 Python 程序的电脑**上的麦克风。它不会直接采集网页访问者的麦克风。

```python
from speechio import LocalModelConfig, TranscriptionOptions, transcribe_microphone

asr_config = LocalModelConfig(adapter="whisper", model_path="models/whisper-base")
try:
    for result in transcribe_microphone(
        config=asr_config,
        options=TranscriptionOptions(language="zh"),
        on_partial_text=lambda text: print("实时文字：", text),
    ):
        print("最终文字：", result.text)
except KeyboardInterrupt:
    pass
```

程序会检测说话与停顿，分段返回最终结果。`on_partial_text` 接收可能继续变化的实时文字，迭代器返回每段最终 `Transcript`。上例按 Ctrl+C 退出，不用于演示保留退出瞬间的尾部结果。

需要由应用按钮控制停止时，使用 `MicrophoneTranscriber(asr_config, ...)` 保留对象：

1. 调用 `start()`，并持续迭代该对象读取结果；使用 `with` 也会启动采集。
2. 应用停止录音时调用 `end_input()`。它停止接收新输入，并处理已接收的音频。
3. 继续读取迭代器，直到结束，再关闭对象。直接 `close()` 不是取得尾部结果的方式。

常用可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `options` | `None` | 使用默认识别选项 |
| `microphone` | `None` | 使用默认输入设备；可传 `MicrophoneConfig(device_index=...)` |
| `segmentation` | `None` | 使用默认分段；可传 `SegmentationConfig(post_speech_silence=0.8)`，停顿 0.8 秒后分段 |
| `on_partial_text` | `None` | 接收实时文字的回调 |
| `realtime` | `True` | 是否启用实时识别 |
| `completion_mode` | `"auto"` | 原生流式识别默认采用流式最终结果 |
| `drain_timeout_seconds` | `10.0` | 停止输入后的结果排空等待设置 |

用 `list_input_devices()` 查看麦克风设备。云端麦克风识别可选择 `qwen-audio-3.1-asr-flash-streaming`。短音频型号没有实时识别能力，使用它时应设置 `realtime=False`，在每段结束后识别。

Whisper 的实时文字通过重复识别已经收到的音频生成，分段结束后还会识别完整片段。原生流式模型则持续接收新增音频，`completion_mode="auto"` 使用其最终结果；显式选择 `"final"` 才要求结束后再识别完整片段。

### 应用自己提供连续音频

浏览器、网络连接或其他录音设备产生的音频可以交给 `STTRuntime`。运行对象持有模型，会话负责一次或多次录音分段。

下面演示应用明确知道一段话何时开始、何时结束的情况。`chunks` 是调用方提供的一段 PCM16 音频块，不含文件头：

```python
from speechio import AudioFormat, SessionConfig, STTError, STTRuntime

def recognize_turn(chunks, asr_config):
    with STTRuntime(asr_config) as runtime:
        session = runtime.create_session(SessionConfig(
            format=AudioFormat(sample_rate=16000, channels=1, sample_type="int16"),
            max_audio_samples=16000 * 60,
        ))
        try:
            session.start()
            for chunk in chunks:
                session.feed_audio(chunk)
            session.end_turn()
            transcript = None
            while True:
                event = session.next_event(timeout=1.0)
                if event is None:
                    continue
                if event.kind == "final":
                    transcript = event.result
                elif event.kind == "failed":
                    raise STTError(event.error.code, event.error.operation, event.error.message)
                elif event.kind == "completed":
                    return transcript
        finally:
            session.close()
```

`end_turn()` 结束该段输入，不代表识别已经完成。随后读取 `final` 和 `completed` 事件。下一段可以再次 `start()`；多段并行提交时，用事件的 `turn` 区分归属。上例只处理一段，不演示并发提交。

`SessionConfig.realtime` 默认 `False`；`max_audio_samples` 默认 `480000`，在 16 kHz 下相当于每段 30 秒，上例明确改为 60 秒。需要边输入边接收云端实时文字时，在 `STTRuntime` 传入 `RealtimeConfig(mode="streaming")`，并设置 `SessionConfig(realtime=True)`；应用应在输入音频期间同时读取事件。

需要 speechio 根据停顿自动分段时，使用 `runtime.create_auto_session(...)`，通过 `feed_audio()` 送入音频，通过 `end_input()` 结束输入。无需为了识别一个普通文件而创建这些会话对象。

## 5. 将文字转换为音频

### 查询能力与选择音色

创建 `Synthesizer(config)` 时就选定了模型。随后可以查询它的能力和音色，不需要再次传入模型配置，也不需要先调用 `prepare()`。

```python
from speechio import LocalModelConfig, SynthesisOptions, Synthesizer

tts = Synthesizer(LocalModelConfig(
    adapter="kokoro",
    model_path="models/kokoro-v1.1-zh",
))
try:
    print("支持的设备：", tts.capabilities.devices)
    print("支持追加文字：", tts.capabilities.text_streaming)
    voices = tts.list_voices()
    for voice in voices:
        print(voice.id, voice.name, voice.languages)

    # 本例选择中文音色；实际应用可以把列表交给用户选择。
    selected = next((v for v in voices if v.languages and "zh" in v.languages), None)
    if selected is not None:
        tts.synthesize(
            "你好，欢迎使用语音合成。",
            options=SynthesisOptions(voice=selected.id, language="zh"),
        ).save("hello.wav")
finally:
    tts.close()
```

`capabilities` 不联网、不下载、不加载模型权重；本地适配器可能读取模型的小型配置文件。查询结果描述适配器与所选模型支持的功能，不检查当前电脑是否装好依赖、模型文件是否齐全、GPU 是否可用，或密钥是否有效。进入 `with Synthesizer(...)` 会执行 `prepare()`；只查询时使用上例的创建和关闭方式。

`Transcriber.capabilities` 和 `STTRuntime.capabilities` 返回识别能力；`Synthesizer.capabilities` 返回合成能力。已有的整段、流式、采样率、时间戳等字段保持原有含义，新增字段如下：

| 字段 | 适用对象 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `languages` | STT、TTS | `None` | 已声明的语言值；可传给该适配器的语言参数 |
| `devices` | STT、TTS | `None` | 适配器支持的执行设备，如 `("cpu", "gpu")`；云端为 `()`，由服务安排设备 |
| `voice_required` | TTS | `None` | 是否要求指定音色；直接提供参考音频时，由克隆步骤提供音色 |
| `default_voice` | TTS | `None` | 适配器声明的默认音色 ID；speechio 不自动选择列表第一项 |
| `speed_range` | TTS | `None` | 语速倍率的闭区间，如 `(0.5, 2.0)` |
| `cloning_requirements` | TTS | `None` | `VoiceCloningRequirements`，描述参考录音要求 |

**这些新增说明字段中的 `None` 表示未知或未声明，不能理解为不支持、无限制或可随意省略。** 判断是否支持克隆仍看 `voice_cloning`；判断语速控制仍看 `speed_control`。当前三个内置 TTS 适配器均要求提供音色，未声明默认音色。

语言值遵循对应适配器的接口。例如，Kokoro 返回 `zh`、`en`、`en-gb`；Qwen Base 从所选模型的 `config.json` 读取语言，中文和英文使用 `zh`、`en`，其他语言使用模型接受的名称，如 `japanese`。不能确定当前本地识别模型的语言范围时，返回 `None`。百炼 TTS 的语言还取决于音色，应查看具体音色说明。

`VoiceCloningRequirements` 的四个字段均默认 `None`：`min_audio_seconds`、`max_audio_seconds` 表示参考录音时长范围，`min_sample_rate` 表示最低采样率，`reference_text` 为 `"required"`、`"optional"` 或 `"unsupported"`，分别表示参考文字必填、可选、不接受。它描述当前适配器的要求；提供 WAV、M4A 等文件时，speechio 会先解码，不能把服务端接收 WAV 理解为调用方只能提供 WAV。

### 音色列表返回什么

`list_voices(*, include_account=False)` 返回只读 `VoiceInfo` 对象组成的元组。默认只读取本地信息，不联网：

| 适配器 | 返回内容 |
| --- | --- |
| Kokoro | 当前模型 `voices/` 目录中的音色文件；不加载文件里的权重。目录无法读取时报错 |
| Qwen3-TTS Base | `()`；Base 模型没有预设音色，需要使用克隆得到的 `Voice` |
| 百炼 | 随 speechio 发布的官方预设音色目录，按当前模型和服务地域筛选 |

每个 `VoiceInfo` 包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 直接传给 `SynthesisOptions(voice=...)` 的音色 ID |
| `name` | 展示名称；没有名称时使用 ID |
| `languages` | 已知语言元组；`None` 表示未声明完整语言范围 |
| `instruction_control` | 是否支持风格指令；`None` 表示未知 |
| `text_timestamps` | 支持的文字时间信息级别，如 `("sentence", "word")`；`()` 表示不支持，`None` 表示未知 |

后面三个字段默认 `None`。音色是否可用与模型、地域和服务状态有关；列举本地文件不等于验证文件完整性。音色列表也不限制手动输入 ID：供应商新增音色后，即使 speechio 的内置目录尚未更新，仍可直接传入有效 ID 合成。

百炼默认配置对应北京；`dashscope-intl.aliyuncs.com` 对应新加坡。两地的业务空间专属域名也会按地域筛选。自定义代理域名无法确定地域时，`list_voices()` 报 `CAPABILITY_UNSUPPORTED`，不会猜测可用音色；已有指定 ID 合成的方式不受影响。目录依据[官方音色说明](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list)，核对日期为 2026-10-29，运行时不抓取该网页。

如果还要查询当前密钥可访问的云端已创建音色，显式传入 `include_account=True`：

```python
import os
from speechio import RemoteModelConfig, Synthesizer

tts = Synthesizer(RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-tts-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
))
try:
    voices = tts.list_voices(include_account=True)
    for voice in voices:
        print(voice.id, voice.name)
finally:
    tts.close()
```

结果是一个统一列表，包含预设音色和属于当前模型、状态可用的云端已创建音色。查询使用当前配置的服务地址和密钥，内部处理分页及模型核对，不创建或删除音色，也不发起语音合成。权限错误、网络错误或响应错误会抛出 `TTSError`，不会伪装成空列表或只返回预设音色。云端没有已创建音色时，正常返回预设列表。

本地适配器不访问云端，`include_account=True` 不改变结果。应用自行保存的 `.voice` 文件不在此列表中；应用决定保存位置，通过 `Voice.load()` 读取并传给合成接口。没有实现音色查询的自定义适配器会报 `CAPABILITY_UNSUPPORTED`，其原有合成方式仍可使用。

### 完整生成

以下示例要求已安装 `tts-kokoro`，并下载 Kokoro 模型及音色：

```python
from speechio import LocalModelConfig, SynthesisOptions, Synthesizer

tts_config = LocalModelConfig(adapter="kokoro", model_path="models/kokoro-v1.1-zh")
with Synthesizer(tts_config) as synthesizer:
    audio = synthesizer.synthesize(
        "你好，欢迎开始今天的练习。",
        options=SynthesisOptions(voice="zf_001", language="zh"),
    )
    audio.save("hello.wav")
```

Kokoro 英文可使用 `voice="af_maple", language="en"`；`bf_vale` 对应英式英语。对应音色文件应位于模型目录的 `voices/` 中。音色名称属于所选模型，不能跨模型直接使用。

`Synthesizer` 在多次合成间复用模型。只调用一次时，也可以用 `synthesize(text, *, config, options=None, reference_audio=None, reference_text=None)`，由函数管理资源。

| `SynthesisOptions` 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `voice` | `None` | 模型的预设音色字符串，或声音克隆返回的 `Voice` |
| `language` | `None` | 语言提示，如 `"zh"`、`"en"` |
| `speed` | `1.0` | 语速倍率；改变默认值需要模型支持 |
| `instruction` | `None` | 风格指令；需要模型支持 |
| `sample_rate` | `None` | 输出采样率；省略时使用模型默认值 |

这些字段在配置类型上均可省略，但所选模型可以要求音色。当前 Kokoro 需要预设音色名称，Qwen Base 需要克隆的 `Voice`，百炼需要可用的音色 ID 或 `Voice`。`voice=None` 不表示随机选择音色；只有提供原生默认音色的适配器才能省略它。

完整合成返回 `AudioResult`：`data` 为单声道小端 PCM16 字节，`format` 描述采样率等格式，`duration_seconds` 表示时长。`save(path, overwrite=False)` 保存 WAV；目录应已存在，默认不覆盖已有文件。speechio 不自动播放音频，应用可以保存后播放，也可以把采样数据交给播放器。

### 一次传入完整文本，逐块接收音频

下面仍使用 Kokoro。`stream()` 的 `text` 和 `options` 都需要提供：

```python
import wave
from speechio import LocalModelConfig, SynthesisOptions, Synthesizer

tts_config = LocalModelConfig(adapter="kokoro", model_path="models/kokoro-v1.1-zh")
with Synthesizer(tts_config) as synthesizer:
    with synthesizer.stream(
        "第一句话。接下来是第二句话。",
        options=SynthesisOptions(voice="zf_001", language="zh"),
    ) as stream:
        with wave.open("stream.wav", "wb") as output:
            output.setnchannels(stream.format.channels)
            output.setsampwidth(2)
            output.setframerate(stream.format.sample_rate)
            for chunk in stream:
                output.writeframesraw(chunk.data)
```

每次迭代返回 `AudioChunk`，包含 `data` 和 `format`。音频块是 PCM 数据，不是各自独立的 WAV 或 MP3 文件。上例用 Python 的 `wave` 写成一个 WAV；该写法会覆盖同名文件。

Kokoro 每次先生成一个文本片段的音频，再返回这一段。Qwen Base 当前适配器只提供完整生成，不能调用 `stream()`。

### 持续追加文字，同时接收音频

当应用的回复文字不断产生时，可以使用支持追加文本的模型。百炼提供此能力。`open_stream(options=...)` 创建流，`send_text()` 追加文字，`finish_input()` 表示文字发送完毕；应用仍应继续接收剩余音频。

下面的函数接收第 3 节方式创建的百炼 TTS 配置、可用的音色，以及应用提供的文字片段迭代器：

```python
import wave
from concurrent.futures import ThreadPoolExecutor
from speechio import SynthesisOptions, Synthesizer

def save_stream(stream, path):
    with wave.open(path, "wb") as output:
        output.setnchannels(stream.format.channels)
        output.setsampwidth(2)
        output.setframerate(stream.format.sample_rate)
        for chunk in stream:
            output.writeframesraw(chunk.data)

def synthesize_text_parts(text_parts, tts_config, voice, output_path):
    with Synthesizer(tts_config) as synthesizer:
        with synthesizer.open_stream(options=SynthesisOptions(voice=voice)) as stream:
            with ThreadPoolExecutor(max_workers=1) as executor:
                receiver = executor.submit(save_stream, stream, output_path)
                try:
                    for text in text_parts:
                        stream.send_text(text)
                    stream.finish_input()
                    receiver.result()
                except BaseException:
                    stream.cancel()
                    raise
```

输入和输出需要同时推进。不能先发送很长的文字，等全部合成完再读取音频。`Synthesizer.max_buffer_seconds` 默认 `10.0`，限制尚未取走的音频；长期不读取可能触发 `BUFFER_EXCEEDED`。

| 适配器 | 完整生成 | 完整文本逐块输出 | 持续追加文本 | 声音克隆 |
| --- | --- | --- | --- | --- |
| Qwen Base 0.6B / 1.7B | 支持 | 不支持 | 不支持 | 支持 |
| Kokoro 82M v1.1-zh | 支持 | 按片段生成 | 不支持 | 不支持 |
| 百炼 Qwen TTS | 支持 | 支持 | 支持 | 支持 |

调用前可读取 `synthesizer.capabilities`，例如 `voice_cloning`、`audio_streaming`、`text_streaming`。不支持的功能会报错，不会自动切换模型。

### 获取文字对应的音频时间

需要让字幕跟随语音播放时，在 `SynthesisOptions` 中设置 `text_timestamps`。这不改变文字的发送方式，也不需要把每句话拆成单独请求。

| 值 | 返回内容 |
| --- | --- |
| `None`（默认） | 只获取音频，保留原有用法 |
| `"sentence"` | 每段文字对应的音频起止位置 |
| `"word"` | 段落位置，以及供应商或本地模型提供的字词位置 |

目前的支持范围如下：

| 模型 | 句段时间 `sentence` | 字词时间 `word` | 返回时机 |
| --- | --- | --- | --- |
| 百炼 `qwen-audio-3.1-tts-flash` | 支持 | 取决于音色 | 句子开始信息先于音频；字词信息可能随后更新 |
| Kokoro 82M v1.1-zh，中文 `zh` | 支持 | 不支持 | 每个片段生成完后，先返回该段时间，再返回该段音频 |
| Kokoro 82M v1.1-zh，英文 `en` / `en-gb` | 支持 | 支持，由模型预测的发音时长换算 | 每个片段生成完后，先返回该段及词时间，再返回该段音频 |
| Qwen3-TTS Base 0.6B / 1.7B | 不支持 | 不支持 | — |

调用前可查看 `synthesizer.capabilities.text_timestamps`；空元组表示不支持。该字段列出模型支持的级别，不保证所有语言和音色组合都支持。例如，Kokoro 声明 `("sentence", "word")`，但中文请求 `word` 仍会报 `CAPABILITY_UNSUPPORTED`，不会自动降为句段时间。Kokoro 音色列表中的时间能力按该音色的默认语言列出；显式指定 `language` 时，以本次语言为准。

使用 `stream.read_event(timeout=None)` 读取一个事件，或使用 `stream.events()` 逐个读取。返回值有两种：

- `AudioChunk`：PCM 音频块，按返回顺序交给播放器。
- `TextTiming`：某段文字及其在输出音频中的位置。

`TextTiming` 的字段如下。所有时间均为秒，**从本次输出音频开头计算**，不是网络接收时间或播放器启动时间。

| 字段 | 含义 |
| --- | --- |
| `segment_index` | 本次任务内从 0 开始的段编号，由供应商分句或本地模型的生成片段决定，与 `send_text()` 次数无关 |
| `text` | 该段的文字 |
| `start_seconds` | 该段音频的开始位置 |
| `end_seconds` | 该段音频的结束位置；尚未结束时为 `None` |
| `words` | `WordTiming` 元组；未请求、尚未返回或供应商未提供时为 `None` |

`WordTiming` 包含 `text`、`start_seconds`、`end_seconds`。同一个 `segment_index` 可以多次返回，应用用新事件更新该段信息。句段范围按实际 PCM 长度确定，可能包含停顿；字词范围来自供应商数据或本地模型的发音时长，不按文字长度均分。

下面示例接收已经创建的百炼 `tts_config`。`on_event` 由应用实现，将音频交给播放器，将文字时间交给字幕逻辑：

```python
from concurrent.futures import ThreadPoolExecutor
from speechio import SynthesisOptions, Synthesizer

def synthesize_with_timings(text_parts, tts_config, on_event):
    def receive(stream):
        try:
            for event in stream.events():
                on_event(event)
        except BaseException:
            stream.cancel()
            raise

    with Synthesizer(tts_config) as synthesizer:
        options = SynthesisOptions(
            voice="yuxiaoyun_v3.1",  # 北京地域的预设中文音色
            text_timestamps="word",
        )
        with synthesizer.open_stream(options=options) as stream:
            with ThreadPoolExecutor(max_workers=1) as executor:
                receiver = executor.submit(receive, stream)
                try:
                    for text in text_parts:
                        stream.send_text(text)
                    stream.finish_input()
                    receiver.result()
                except BaseException:
                    stream.cancel()
                    raise
```

输入和读取并行进行；`on_event` 不应等待音频实际播放完再返回，否则会阻塞后续读取。应用根据播放器实际播放到的位置显示字幕，而不是收到文字事件就立即显示。

百炼一句话的开始事件先于对应音频交付。字级信息可能随合成逐步补全，也可能到句末才完整返回。因此，无缓冲立即播放时不能保证每个字的时间都提前可用；需要精确逐字显示的应用应自行安排播放缓冲。speechio 不延迟播放、不补造缺失的时间，也不额外调用识别模型。

音色支持情况以[百炼音色列表](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list)为准。官方声明支持复刻音色；本次实际验证使用预设音色 `yuxiaoyun_v3.1`，未验证克隆音色。其他地域、音色或模型不能据此视为已实测。

Kokoro 可以一次传入整段文字，通过 `stream()` 逐段取得时间和音频。它仍不支持 `open_stream()` 持续追加文字。以下示例使用已下载的本地模型：

```python
from speechio import LocalModelConfig, SynthesisOptions, Synthesizer

config = LocalModelConfig(adapter="kokoro", model_path="models/kokoro-v1.1-zh")
options = SynthesisOptions(voice="af_maple", language="en", text_timestamps="word")

with Synthesizer(config) as synthesizer:
    with synthesizer.stream("Hello! Welcome to speechio.", options=options) as stream:
        for event in stream.events():
            on_event(event)  # 应用分别处理 AudioChunk 和 TextTiming
```

中文可改用 `voice="zf_001", language="zh", text_timestamps="sentence"`。在异步应用中使用 `await synthesizer.stream_async(...)` 创建流，再循环 `await stream.read_event_async()`，直到返回 `None`。

Kokoro 的“段”是实际生成的文本片段，长句因模型容量拆分时会返回多个 `TextTiming`。每段结束位置以实际采样数计算，后续段和词的时间累计到本次输出音频开头。每段生成完即可提供对应信息，无须等待全文完成，但第一段推理期间仍需等待。

英文词时间来自同次合成的预测发音时长，反映模型生成安排，不是人工校准的精确发音边界。词文本保留原文中的数字、缩写和标点；标点也可能单独返回一项。无法把原文、音素和时长对应起来时会报 `PROTOCOL_ERROR`，不会补造词时间。只需分段字幕时可使用 `sentence`。

`stream(text, options=...)` 同样可以读取事件。完整生成 `synthesize()` 将最终段信息放在 `AudioResult.text_timings` 中，默认是空元组；它收集同一次合成的结果，不再发起第二次合成。`AudioResult.save()` 仍只保存 WAV 音频，文字时间由应用另行保存。

原有的 `read()` 和 `for chunk in stream` 继续只返回音频，即使开启了时间信息也会跳过文字事件。同一个流只能选择一种读取方式，不能中途切换或同时使用多个读取方。`read_event()` 成功读尽时返回 `None`，超时抛出 `READ_TIMEOUT`，不代表合成结束。取消或失败会清除待读取事件；文字事件也受队列容量限制，长期不读取会报 `BUFFER_EXCEEDED`。

## 6. 克隆声音与复用音色

### 独立克隆并保存

声音克隆需要指定支持克隆的模型。顶层函数通过 `config` 选择模型，实例方法使用创建 `Synthesizer` 时的模型。

以下示例使用 Qwen Base 0.6B。准备参考录音 `reference.wav`，并将 `reference_text` 替换为录音原文；没有原文时可省略它：

```python
from speechio import LocalModelConfig, SynthesisOptions, Synthesizer, Voice

clone_config = LocalModelConfig(
    adapter="qwen_tts",
    model_path="models/qwen3-tts-0.6b-base",
    device="cpu",
)
with Synthesizer(clone_config) as synthesizer:
    voice = synthesizer.clone_voice(
        "reference.wav",
        reference_text="这里填写参考录音中实际说出的文字。",
    )
    voice.save("speaker.voice")

# 以下部分可以在另一次启动程序时执行。
restored_voice = Voice.load("speaker.voice")
with Synthesizer(clone_config) as synthesizer:
    audio = synthesizer.synthesize(
        "这是新生成的内容，与参考录音的文字不同。",
        options=SynthesisOptions(voice=restored_voice, language="zh"),
    )
    audio.save("cloned-speech.wav")
```

单次克隆也可以使用 `clone_voice(reference_audio, *, config, reference_text=None)`。`reference_audio` 和 `config` 必填，参考音频接受文件路径或 `AudioResult`。

Qwen 省略 `reference_text` 时只提取声音特征；填写准确原文时还会使用参考录音的发音内容。选择 1.7B 时，把模型路径改为对应目录。0.6B 和 1.7B 的克隆结果不互用，更换规格后需要用新模型重新克隆。

### 克隆结果是什么

`Voice` 是用于合成的声音数据，不是可播放的录音。`Voice.save(path, overwrite=False)` 将它保存成 JSON 形式的数据文件，扩展名可使用 `.voice`；目录应已存在，默认不覆盖。之后通过 `Voice.load(path)` 读取。

- 本地 Qwen 的 `Voice` 保存声音特征和模型需要的参考信息。再次使用不需要原参考录音，但仍需要兼容的模型文件。
- 云端 `Voice` 保存服务端音色 ID、模型和服务地址，不包含密钥或原录音。使用时仍需联网并提供有权限的密钥；删除服务端音色后，本地文件不能恢复它。

`Voice` 不是跨模型通用格式。应用负责保存位置、用户自定义的音色名称、用户归属和本地 `.voice` 文件列表。查询模型预设音色或云端已创建音色使用 `Synthesizer.list_voices()`。

### 使用百炼克隆

以下例子只克隆并保存，不生成语音：

```python
import os
from speechio import RemoteModelConfig, clone_voice

clone_config = RemoteModelConfig(
    provider="bailian",
    model="qwen-audio-3.1-tts-flash",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    timeout_seconds=60,
)
voice = clone_voice("reference.wav", config=clone_config)
voice.save("cloud-speaker.voice")
```

该适配器上传参考录音，等音色可用后返回 `Voice`。参考录音要求 5–60 秒、采样率至少 16 kHz；转换后的单声道 PCM16 WAV 不超过 10 MiB。百炼克隆不接收 `reference_text`，不要传入该字段。

加载保存的 `Voice` 后，仍通过 `SynthesisOptions(voice=voice)` 合成。配置应使用相同的模型和服务地址。创建成功后若状态查询失败，异常的 `__notes__` 会保留已返回的音色 ID；超时不一定表示服务端没有创建音色。speechio 不自动重试创建，也不在关闭对象时删除云端音色。

### 提供参考录音，直接生成语音

只需要一次生成时，可以把参考录音传给完整合成接口。下面的 `clone_config` 使用本节任一本地或云端配置：

```python
from speechio import SynthesisOptions, synthesize

audio = synthesize(
    "这段文字使用参考录音的声音朗读。",
    config=clone_config,
    reference_audio="reference.wav",
    options=SynthesisOptions(language="zh"),
)
audio.save("speech.wav")
```

这个入口复用独立克隆流程，然后用得到的 `Voice` 合成。每次提供参考录音都会执行克隆，不自动保存音色文件。需要多次使用同一声音时，先独立克隆并保存，再传入 `Voice`，避免重复克隆。

`reference_audio` 与 `options.voice` 不能同时提供；没有参考音频时也不能单独提供 `reference_text`。克隆失败则不继续合成。云端调用产生的音色不会在合成结束后自动删除。

`stream()` 和 `open_stream()` 不接收参考录音。使用流式合成时，先独立克隆，再传入 `Voice`，并选择支持流式合成的模型。

## 7. 在应用中组合识别与合成

对话应用可以按以下顺序使用 speechio：

```text
用户录音 → speechio 识别 → 应用生成回复文字 → speechio 合成 → 应用播放
```

下面用函数参数区分语音能力与应用逻辑。`make_reply` 由应用实现，接收用户文字并返回回复文字；它可以调用应用选择的对话模型：

```python
from speechio import SynthesisOptions, Synthesizer, Transcriber

def reply_to_recording(audio_path, transcriber, synthesizer, voice, make_reply):
    transcript = transcriber.transcribe_file(audio_path)
    reply_text = make_reply(transcript.text)
    reply_audio = synthesizer.synthesize(
        reply_text,
        options=SynthesisOptions(voice=voice),
    )
    return transcript.text, reply_text, reply_audio
```

应用启动时创建 `Transcriber(asr_config)` 和 `Synthesizer(tts_config)`，在多次对话中复用，退出时关闭。`voice` 使用对应合成模型的音色字符串或 `Voice`。应用决定何时录音、何时发起下一轮、是否中断播放，以及如何保存历史。

浏览器应用应由前端采集和播放音频，由 Python 服务调用 speechio。页面布局、HTTP/WebSocket 接口和会话用户管理不属于核心包。speechio 的调用是同步接口；异步服务应通过工作线程等方式执行阻塞调用，不能直接给这些方法加 `await`。

## 8. 错误、停止与资源释放

用 `with` 管理处理对象与流，确保正常退出和异常退出都会发起清理。模型加载、网络请求和合成可能失败：识别错误为 `STTError`，合成及克隆错误为 `TTSError`；两者都属于 `SpeechioError`。

```python
from speechio import SpeechioError, transcribe_audio

def try_transcribe(path, asr_config):
    try:
        return transcribe_audio(path, config=asr_config)
    except SpeechioError as error:
        print(error.code, error.operation, error.message)
        raise
```

先根据 `code` 判断错误类别，再决定是否修改配置或提示用户。模型文件缺失、设备不可用、功能不支持和服务失败的处理不同。speechio 不会为了让调用成功而自动切换模型或设备，也不自动重试收费任务。

停止输入与取消任务含义不同：

| 操作 | 含义 |
| --- | --- |
| 麦克风或自动会话的 `end_input()` | 不再接收新音频，继续处理已有输入 |
| 手动会话的 `end_turn()` | 结束当前片段，等待最终结果 |
| 合成流的 `finish_input()` | 文字已发送完毕，继续接收剩余音频 |
| 会话的 `cancel_turn()` | 取消指定或当前片段 |
| 合成流的 `cancel()` | 停止交付，并请求取消后续工作 |
| `close()` | 释放对象资源；不是获取未读结果的方法 |

取消不能强行终止已经进入本地模型的一次计算。`Transcriber.close(timeout=5.0)`、`Synthesizer.close(timeout=5.0)` 和 `STTRuntime.close(timeout=5.0)` 限制的是本次等待时间。`timeout=0` 发起关闭后立即查询状态。

提供关闭状态的接口用 `ok` 表示已完成、`pending` 表示仍在退出、`failed` 表示清理失败。`pending` 不表示模型或线程已经释放。拥有者会继续后台清理，可以再次调用 `close()` 确认状态。使用上下文管理器时，未正常完成的关闭可能作为异常报告；已有正文异常不会被关闭异常替换。

`Transcriber`、`Synthesizer` 等单任务对象忙时会拒绝重叠任务，不会替应用无限排队。应用应复用对象并控制调用顺序；需要并发时应结合内存、显存和服务额度安排独立对象或使用支持调度的识别运行对象。

### 在异步应用中调用

异步接口适合 FastAPI、WebSocket 等使用 `asyncio` 的应用。在等待识别或合成时，事件循环可以继续处理其他连接。它不会提高模型本身的推理速度，也不让同一个单任务对象同时执行多个任务。原有同步接口仍可使用。

异步方法使用与对应同步方法相同的配置、选项和结果类型，方法名增加 `_async`。新增任务参数 `cancellation` 默认 `None`；不传时，异步入口会为这次调用创建取消令牌。

| 入口 | 异步方法 |
| --- | --- |
| 一次性函数 | `transcribe_audio_async()`、`transcribe_url_async()`、`synthesize_async()`、`clone_voice_async()` |
| `Transcriber` | `prepare_async()`、`transcribe_async()`、`transcribe_chunks_async()`、`transcribe_file_async()`、`transcribe_url_async()` |
| `Synthesizer` | `prepare_async()`、`synthesize_async()`、`clone_voice_async()`、`list_voices_async()`、`stream_async()`、`open_stream_async()` |
| `STTRuntime` | `prepare_async()`、`create_auto_session_async()`、`create_auto_session_from_detector_async()` |
| 识别会话 | `feed_audio_async(chunk)`、`next_event_async(timeout=None)`；自动会话另有 `end_input_async()` |
| 合成流 | `send_text_async(text)`、`finish_input_async()`、`read_async(timeout=None)`、`read_event_async(timeout=None)` |
| 关闭 | `Transcriber`、`Synthesizer`、`STTRuntime`、`SynthesisStream` 都提供 `close_async(timeout=5.0)` 和 `async with` |

`transcribe_chunks_async()` 接收有限的同步可迭代音频块；网络持续产生音频时，使用会话的 `feed_audio_async()`。同一会话逐块等待处理结果，不要为每个块创建一个独立任务。`start()`、`end_turn()`、`cancel_turn()` 等短控制操作继续同步调用。

`next_event_async()` 默认等待，超时返回 `None`；同步 `next_event()` 默认不等待。合成流的读取超时仍抛出 `TTSError(code="READ_TIMEOUT")`，成功读尽才返回 `None`。同一合成流不能混用只读音频与读取文字时间事件两种方式。

一次性异步识别示例：

```python
from speechio import transcribe_audio_async

async def recognize_file(path, asr_config):
    result = await transcribe_audio_async(path, config=asr_config)
    return result.text
```

持续追加文字并接收音频时，发送与读取应同时进行。下例的 `texts` 是应用提供的异步文字来源，`send_audio` 是应用提供的异步音频发送函数：

```python
import asyncio
from speechio import Synthesizer, SynthesisOptions

async def speak(texts, send_audio, tts_config, voice):
    async with Synthesizer(tts_config) as tts:
        async with await tts.open_stream_async(
            options=SynthesisOptions(voice=voice)
        ) as stream:
            async def send_text():
                async for text in texts:
                    await stream.send_text_async(text)
                await stream.finish_input_async()

            async def receive_audio():
                async for chunk in stream:
                    await send_audio(chunk.data, chunk.format)

            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(send_text())
                tasks.create_task(receive_audio())
```

该例要求模型支持持续追加文字。需要文字与音频时间对应信息时，在选项中请求相应精度，并将读取改为循环 `await stream.read_event_async()`；具体模型限制仍见前面的时间信息说明。

### 取消一个任务

`CancellationToken` 表示一次任务的取消请求，不代表模型已经停止。每个独立任务创建一个新令牌；不要复用已经取消的令牌。

| 成员 | 含义 |
| --- | --- |
| `CancellationToken()` | 创建令牌，无必填参数 |
| `cancel_requested` | 只读布尔值，初始为 `False` |
| `cancel()` | 可跨线程调用，重复调用无额外效果；只发出请求，不等待执行退出 |
| `wait_cancelled(timeout=None)` | 给同步适配器使用；等到请求返回 `True`，等待超时返回 `False` |

同步与异步的准备、整段识别、合成、克隆、音色查询、流创建方法都可传 `cancellation=token`。`synthesize(reference_audio=...)` 的克隆与合成共用同一令牌；在克隆结束后取消，不再开始合成。创建合成流时传入的令牌在返回后仍控制整条流。

```python
import asyncio
from speechio import CancellationToken, TTSError

async def generate_with_stop(tts, text, stop_requested):
    # stop_requested 是应用在用户点击“停止”时设置的 asyncio.Event。
    token = CancellationToken()

    async def watch_stop():
        await stop_requested.wait()
        token.cancel()

    watcher = asyncio.create_task(watch_stop())
    try:
        return await tts.synthesize_async(text, cancellation=token)
    except TTSError as error:
        if error.code != "CANCELLED":
            raise
        return None
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
```

也可以取消 Python 的任务对象：`task.cancel()`，然后 `await task` 会抛出 `asyncio.CancelledError`。对于识别、合成等任务入口，这也会向底层发出取消请求。显式 `token.cancel()` 则通过 `STTError` 或 `TTSError` 的 `CANCELLED` 错误报告；同步调用或不支持中断的底层调用可能要等到下一处取消检查才能返回。

不同操作的取消范围如下：

| 被取消的等待 | 影响 |
| --- | --- |
| `next_event_async()`、`read_async()`、`read_event_async()` | 只取消本次等待；任务继续，之后仍能读取结果 |
| `send_text_async()`、`finish_input_async()` | 已发送的文字无法撤回，因此取消整条合成流 |
| 已开始的 `feed_audio_async()`、`end_input_async()` | 关闭本次识别会话，不关闭运行时或其他会话 |
| `session.cancel_turn(turn)` | 取消指定轮次，向其识别任务传递请求；其他轮次不受影响 |
| `stream.cancel()` | 丢弃尚未交付的音频，包括已生成但未读取的音频；已交给应用播放的内容由应用停止 |

本地模型进入原生计算后通常不能强制中断。取消后结果不再交付，但对象继续占用模型，直到实际调用退出。`Transcriber.busy`、`Synthesizer.busy` 可查询占用状态；`wait_idle(timeout=5.0)` 和 `await wait_idle_async(timeout=5.0)` 等待可复用，返回 `True` 表示空闲，`False` 表示等待超时。它们不取消任务、不关闭对象；已关闭的对象不能靠等待重新使用。

百炼 WebSocket 任务通过中断当前连接解除发送或接收等待。普通 HTTP 请求不能保证立即中止，但取消后不会继续下一阶段；长录音会停止轮询和下载结果。已经提交的云端任务可能继续执行或计费。克隆创建已经返回音色 ID 时，取消异常的 `__notes__` 会保留该 ID；speechio 不会自动删除音色。

## 9. 接入自定义模型或服务

适配器是把某个模型自身的调用方式转换为 speechio 契约的 Python 对象。本地与云端都使用这种扩展方式。

适配器仍实现同步方法，由 speechio 的异步入口安排执行，不需要再写一套异步适配器。旧适配器可以继续使用；如果方法增加可选的 `cancellation=None` 参数，speechio 会传入本次任务令牌。适配器可在分块处理、提交远程请求或轮询之间检查 `cancellation.cancel_requested`，并抛出 `STTError("CANCELLED", "transcribe", "任务已取消")` 或对应的 `TTSError`。等待轮询间隔时，可用 `cancellation.wait_cancelled(seconds)`，让等待能被取消唤醒。

未接收该参数的旧适配器只能在返回后由 speechio 丢弃迟到结果。speechio 接入时检查方法签名，不会因为方法抛出 `TypeError` 而重试一次调用。

使用内置适配器时传 `LocalModelConfig` 或 `RemoteModelConfig`。使用自己的适配器时，先完成它自己的配置，再把实例传给 `Transcriber.from_adapter(adapter)` 或 `Synthesizer.from_adapter(adapter)`，不再重复传一份 `config`，也不需要注册到内置模型列表。

### 最小识别适配器

下面接收应用提供的 `recognize` 函数。这个函数负责实际调用模型，接收标准音频数组和语言参数，并返回字符串：

```python
from speechio import STTAdapter, STTCapabilities, Transcript, Transcriber

class MySTTAdapter(STTAdapter):
    """把应用自己的整段识别函数接入 speechio。"""

    def __init__(self, recognize):
        self._recognize = recognize

    @property
    def capabilities(self):
        return STTCapabilities(engine="my-asr", transcription=True)

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        text = self._recognize(audio, language=language)
        return Transcript(text=text)


def recognize_with_adapter(path, recognize):
    with Transcriber.from_adapter(MySTTAdapter(recognize)) as transcriber:
        return transcriber.transcribe_file(path)
```

适配器的 `transcribe()` 收到标准化后的 16 kHz、单声道 `float32` 数组。文件解码和输入格式转换由 speechio 处理。示例没有实时或时间信息能力，所以没有声明它们。

需要原生流式识别时，实现 `create_streaming_session()` 并返回符合 `STTStream` 契约的对象。其 `accept_audio()` 接收新增音频，`get_result()` 返回当前结果，`finish()` 排空输入并返回最终结果。可实现非阻塞、允许与发送接收并发的 `cancel()` 来中断等待，最终资源仍由 `close()` 释放。

### 最小合成适配器

下面的 `generate` 由应用实现，返回指定采样率的单声道小端 PCM16 字节：

```python
from speechio import AudioFormat, AudioResult, Synthesizer, TTSAdapter, TTSCapabilities

class MyTTSAdapter(TTSAdapter):
    """把应用自己的完整合成函数接入 speechio。"""

    def __init__(self, generate, sample_rate=24000):
        self._generate = generate
        self._sample_rate = sample_rate

    @property
    def capabilities(self):
        return TTSCapabilities(default_sample_rate=self._sample_rate, synthesis=True)

    def synthesize(self, text, *, options):
        data = self._generate(text, voice=options.voice)
        return AudioResult(
            data=data,
            format=AudioFormat(sample_rate=self._sample_rate, channels=1, sample_type="int16"),
        )
```

将实例传给 `Synthesizer.from_adapter(MyTTSAdapter(generate))`，即可使用完整合成入口。要支持音频流、追加文本或克隆，分别实现契约中的对应方法，并在 `TTSCapabilities` 中声明。只有完整生成能力时，不应声明流式能力。

自定义适配器可按实际支持情况填写新增能力说明字段。不填写时保持 `None`。要支持音色查询，实现 `list_voices(self, *, include_account=False)`，返回 `VoiceInfo` 序列；查询不能调用 `prepare()` 加载模型。默认只读取本地资料，只有 `include_account=True` 才可查询云端已创建音色。没有账户概念的适配器忽略这个开关。

两个最小示例都包装了外部函数，因此没有自行持有模型资源。实际适配器如果创建模型或连接，应在 `prepare()` 中准备，在 `close()` 中释放。处理对象接管适配器的生命周期；不要把同一实例交给多个独立的处理对象。

识别适配器的 `close()` 返回 `None` 或 `CloseResult(status="ok")` 表示完成。如果资源还在退出，应及时返回 `pending`，后续调用再确认；返回 `failed` 表示最终失败。不要通过吞掉异常来声称释放成功。
