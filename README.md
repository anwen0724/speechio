# realtime-asr

独立实时 ASR 项目。当前提供同步整段识别 SDK、连续会话 SDK、实时识别与稳定文本、自动分句和结束优化、麦克风采集和本地命令行。库是产品主体，命令行只是仓库内的本地工具，不随 wheel 发布。真实评估入口独立放在 `evals/`。

同步 SDK 分块入口在全部输入结束后返回完整结果。连续会话可在上一单元推理期间接收下一单元，通过实时与最终事件消费结果。基本实时支持 Whisper 完整快照和 Nemotron 原生增量。自动会话通过双 VAD 确认录音段，提供前置保护与统一回溯。可选策略提供声学触发、提前推理、动态等待及有词时间证据的句末切段。双实时可分别配置准确与极速角色，保留原文并按锚点合并有限尾部。候选支持固定尾部／整轮快照、手动同轮继续、候选完成和最近整轮修正。SDK 与 CLI 使用相同核心接口；可选本地唤醒已接通。产品范围见 [PRD](docs/PRD.md)，系统设计见 [Architecture](docs/ARCHITECTURE.md)，实际验收见 [STATUS](docs/STATUS.md)。

## 日常启动

当前入口统一为 `python -m apps.cli`，本机参数保存在项目根目录的 `config.toml`。分发包名仍为 realtime-asr，Python 导入名仍为 realtime_asr。

本机已创建配置后，打开 PowerShell：

```powershell
Set-Location C:/all/work/practice/realtime-asr
. .venv/Scripts/Activate.ps1
python -m apps.cli config check
python -m apps.cli mic
```

文件方式使用 `python -m apps.cli file recording.wav`。麦克风与 stdin 会持续运行，Ctrl+C 停止；文件识别完成后退出。

首次使用且没有配置文件时运行 `python -m apps.cli config init`；已有文件不会被覆盖。编辑并保存配置后重新启动命令生效。`python -m apps.cli config show` 显示 mic 的实际配置与来源，`config show file`、`config check stdin` 可选择其他入口。未激活环境时统一使用 `uv run --no-sync python -m apps.cli ...`。

配置中的相对路径以配置文件目录为准，待转写文件路径和命令行路径仍以调用目录为准。临时参数优先于配置且不会改写文件，例如 `python -m apps.cli file recording.wav --device cuda`。布尔开关支持 --no-*，如 `python -m apps.cli mic --no-realtime`。其他目录使用 `python -m apps.cli --config C:/all/work/practice/realtime-asr/config.toml mic`。

完整字段与注释由 `python -m apps.cli config init` 生成到当前目录的 `config.toml`；已有文件不会被覆盖。本机 config.toml 不提交 Git；密钥使用环境变量。config check 不加载模型或占用设备，成功不代表实际推理或识别质量验收通过。正文默认仍不自动保存，命令行可重定向最终输出。

## 代码结构

库位于 `src/realtime_asr/`，按音频、会话、分句、文本、识别引擎和运行时组织，不依赖 `apps/`。`apps/` 下只有本地命令行，不随 wheel 发布。公共错误与结果各保留一个小文件；库不依赖 evals、测试或示例。

```text
src/realtime_asr/              # 库：只包含核心能力
├── __init__.py                # 稳定的公开 SDK 导出
├── api.py                     # transcribe_audio 与 transcribe_stream
├── bundled.py                 # 内置 Silero v6 检测模型路径
├── assets/                    # 检测模型与许可说明，随 wheel 分发
├── errors.py                  # 公共错误与错误记录
├── results.py                 # 识别与关闭结果
├── model_manifests.py         # 固定权重发布包的不可变描述
├── install_sherpa_models.py   # 把固定 Nemotron 权重安装到显式目录
├── audio/
│   ├── conversion.py          # 格式、连续转换、凑帧与标准样本位置
│   ├── files.py               # 标准 WAV 与通用文件有限块解码
│   └── microphone.py          # 输入设备、实际源格式与独立采集
├── session/
│   ├── session.py             # 单元归属、控制、任务与有序事件
│   └── automatic.py           # 持续源输入与待递交自动段
├── segmentation/
│   ├── gate.py                # 前置保护、静音确认、裁剪与切点
│   ├── acoustic.py            # 有界能量包络与声学触发信号
│   └── vad.py                 # WebRTC、Silero 与独立检测状态
├── text/
│   ├── stabilizer.py          # 原文投影、稳定证据与最终差异
│   ├── endpoint.py            # 动态等待与句末锚点策略
│   └── realtime_merge.py      # 双路观察、锚点、内容代次与有限尾部
├── engines/
│   ├── base.py                # 通用整段、流式与能力契约
│   ├── factory.py             # 引擎名到适配器的注册表
│   ├── local.py               # 本地配置、GPU 与模型组装
│   ├── whisper.py             # Whisper 整段与快照适配
│   ├── nemotron.py            # Nemotron 识别器适配
│   └── stream.py              # 原生流输入、解码、结束与关闭
└── runtime/
    ├── transcriber.py          # 同步 SDK 与模型生命周期
    ├── asr.py                 # ASRRuntime、最终调度及容量
    ├── realtime.py            # 角色执行组、周期、容量与原生排空
    ├── tasks.py               # 固定音频、选项及任务归属
    ├── models/                # 显式准备、身份校验及准备阶段记录
    └── __init__.py

apps/cli/                      # 本地命令行工具，不随库发布
├── options.py                 # 选项与命令声明，argparse、配置和模板的唯一来源
├── parser.py                  # 由声明生成 argparse 与跨参数校验
├── settings.py                # config.toml 读取、来源标注与模板生成
├── candidates.py              # 候选请求时钟
├── input.py                   # raw stdin 与可停止管道读取
├── output.py                  # 文本与 JSONL 出口
├── resources.py               # 模型与 GPU 资源命令
├── runner.py                  # 公开 SDK 接入与自有资源关闭
└── __main__.py                # python -m apps.cli

evals/p0/                    # 语料、评分与基线验证
evals/p1/                    # 同步 SDK 真实验收
evals/p2/                    # 连续会话真实验收
evals/p3/                    # 实时联合运行与资源测量
evals/p4/                    # 自动分句与边界真实验收
evals/p5/                    # 麦克风与 CLI 真实验收
evals/module_organization.py # 目录迁移前后的真实回放

examples/                    # 只通过公开 SDK 使用产品
tests/
├── audio/                   # 音频格式、转换与文件
├── cli/                     # 命令行公开行为与外部输入输出
├── session/                 # 单元、控制、归属与自动输入
├── segmentation/            # 检测、分句与边界保护
├── text/                    # 稳定观察、句末语言规则及双路合并
├── engines/                 # 模型适配与原生流
├── runtime/                 # 模型资源与执行调度
└── evals/                   # 评估工具行为
```

`audio` 提供标准音频与位置；`segmentation` 提供检测记录和切点；`session` 决定段归属、封口、取消和事件有效性。`runtime` 管理模型与推理调度，`engines` 处理模型专有调用和原生流。分句模块不提交 ASR 任务，引擎不读取会话缓冲。

`runtime/models` 与 `runtime/native` 是资源支持模块，可以被运行时和引擎使用；这些模块不反向依赖会话、引擎或执行器。会话在显式协调条件内提交固定任务，并通过完成方法接收结果。队列不持有可变音频单元；实时执行器持有流和排空状态。

`text` 只处理观察与文本策略，不持有模型或可变音频。辅助任务复用 `runtime/asr.py` 的最终模型、任务容量和工作线程；会话执行归属检查与左右切分。

评估代码调用正式实现，再补充语料、测量和评分。`evals/p0/engines.py` 的 `EnginePool` 只用于评估，不作为产品运行时。库只包含 `src/realtime_asr/`；`apps/cli` 是本地命令行工具，不随库发布；公开导入以 `src/realtime_asr/__init__.py` 的 `__all__` 为准，低层组件从子模块导入。

## 自定义引擎

`Transcriber(engine=...)` 和 `ASRRuntime(engine=...)` 接受同一引擎契约。调用方可以实现本地模型或远程 API；引擎不需要提供本地路径。包内提供 Whisper、Nemotron 适配，不提供厂商云端适配器。

引擎继承 `BaseTranscriptionEngine` 或实现相同方法。`capabilities` 返回固定的 `EngineCapabilities`；`prepare()` 准备资源，`close()` 释放资源。整段方法 `transcribe(audio, language=None, *, role="final", word_timestamps=False)` 接收 16 kHz 单声道、PCM16 除以 32768 后的 float32，返回 `Transcript`。语言提示原样传入，厂商映射由引擎完成。能力字段分别声明整段、完整快照、原生流式和词时间；核心按能力组装角色。

流式引擎通过 `create_streaming_session(language=None)` 为每个音频单元创建独立流。流实现 `StreamingSession` 的 `accept_audio()`、`get_result()`、`finish()`，可按需实现 `decode()` 和 `close()`。输入只包含新增标准音频；结果是本单元累计文本。`finish()` 必须排空尾部并返回累计结果，不能把厂商的单句结束直接变成核心会话的完成事件。

注入后 SDK 接管引擎生命周期。同一运行时内，最终和实时角色可共用实例，运行时串行调用并只关闭一次；不同实例各自准备和关闭。`RealtimeConfig(engine=other_engine, mode="native")` 配置独立流式角色。最终角色仍需整段能力；只支持流式的引擎可作为独立实时角色。引擎准备失败后，调用方需新建实例和 SDK 对象。

远程适配器的服务地址、连接池和凭据保存在适配器内。自定义引擎使用方式见 [custom_engine.py](examples/custom_engine.py)，详细输入、错误及资源契约见 [P8.0 Spec](docs/specs/engine-and-runtime-boundaries.md)。

## 双实时识别

第二路默认关闭。最终和准确角色可以共享实例，极速角色可以使用独立模型或调用方提供的引擎。SDK 在 `ASRRuntime` 上配置 `ultrafast`，在 `SessionConfig` 上通过 `merge=RealtimeMergeConfig()` 为该会话开启双实时。

当前可准备 Nemotron 3.5 的 1120 ms／80 ms INT8 变体，均使用 sherpa-onnx CPU 路径。最终与准确实时共用 1120 ms，极速使用独立 80 ms。**本机 CPU 对照没有显示延迟或尾部收益，80 ms 路实际更慢，结束等待也增加。** 这些名称表示角色；不要据此把该组合当作性能默认配置。已有单路配置保持。测量及未验收范围见 [P8.1 Plan](docs/plans/p8-1-dual-realtime.md)。

准备函数只在明确调用时下载；推理保持离线。已准备目录只校验，不把另一块长重标为目标变体。基础包之外需要 `nemotron` 依赖。

```python
from realtime_asr.runtime.models.prepare import prepare_nemotron

prepare_nemotron("models/nemotron-1120ms", variant="1120ms", allow_download=True)
prepare_nemotron("models/nemotron-80ms", variant="80ms", allow_download=True)
```

使用已准备的目录：

```python
from realtime_asr import (
    RealtimeConfig, RealtimeMergeConfig, ModelConfig, SessionConfig,
)
from realtime_asr.runtime.asr import ASRRuntime

accurate = ModelConfig(model="models/nemotron-1120ms", engine="nemotron")
fast = ModelConfig(model="models/nemotron-80ms", engine="nemotron")
with ASRRuntime(
    accurate,
    realtime=RealtimeConfig(mode="native"),
    ultrafast=RealtimeConfig(mode="native", model=fast),
) as runtime:
    session = runtime.create_session(
        SessionConfig(realtime=True, merge=RealtimeMergeConfig())
    )
    try:
        session.start()
        session.feed_audio(pcm16_bytes)  # 调用方已有的标准音频。
        session.end_turn()
        while True:
            event = session.next_event(timeout=60)
            if event is None:
                raise TimeoutError("识别等待超时")
            if event.error:
                raise RuntimeError(event.error.message)
            print(event.kind, event.realtime_role, event.merge)
            if event.kind == "completed":
                break
    finally:
        session.close()
```

`event.result` 保留所属角色的原文；`event.merge` 提供两路来源、合并文本、锚点及范围。稳定与结束策略只消费准确路，最终仍识别完整音频。英文按词、汉字按字符对齐，无可信锚点时保持正文，不能强行拼接。

本地 CLI 的实际参数：

```powershell
python -m apps.cli stdin --model models/nemotron-1120ms --engine nemotron --ultrafast-model models/nemotron-80ms --realtime --dual-realtime --output jsonl
```

独立准确模型使用 `--realtime-model`。两路 backend 和 device 可独立指定，缺省沿用最终配置。

可运行文件示例见 [dual_realtime.py](examples/dual_realtime.py)。具体契约见 [P8.1 Spec](docs/specs/dual-realtime.md)。

## 环境与安装

`cli` 不属于库，不随 wheel 安装，也没有对应的 extras。它位于 `apps/cli`，是仓库内的本地使用工具；需要服务的场景由使用者按公开 API 自行实现。

使用 uv 和 64 位 CPython 3.12。开发解释器由 `.python-version` 固定为已安装的 3.12.14，Python 依赖由 `uv.lock` 锁定。以下命令在项目根目录执行：

```powershell
uv sync --locked --extra nemotron --extra vad --extra microphone
uv run --no-sync python -m evals.p0 environment
```

uv 建立根目录 `.venv`，不向 Anaconda 或其他项目环境安装依赖。`uv run` 也应带上所需 extras；省略 extras 会按较小的依赖集合重新同步环境。

基础包只安装 NumPy，后端依赖按 extras 安装。P0 评估通过项目根目录的 `evals/` 运行，使用开发依赖组中的 SciPy 和 psutil，M10 使用 `tests/`。wheel 只包含 `src/realtime_asr/` 的正式模块，不包含评估工具、测试或语料。源码分发包保留评估与测试，供开发验证使用。

## 本地命令行与麦克风

仓库内提供 CLI：`python -m apps.cli`；库安装不提供命令行。帮助不枚举设备或加载模型。麦克风使用 `microphone`，自动分句额外使用 `vad`；文件使用 `soundfile`／`soxr`。各命令均通过已有会话 SDK 识别，模型路径由调用方指定。

以下示例在已经按上文同步环境、准备本地模型后运行：

```powershell
uv run --no-sync python -m apps.cli --help
uv run --no-sync python -m apps.cli devices

# 默认自动分句和基本实时；Ctrl+C 停止，也可指定 --duration。
uv run --no-sync python -m apps.cli mic --model .runtime/models/whisper-tiny --duration 10

# 手动模式将本次采集作为一个完整单元；不要求 VAD。
uv run --no-sync python -m apps.cli mic --model .runtime/models/whisper-tiny --mode manual --duration 5 --output jsonl

# 按文件顺序识别，各文件使用自己的实际格式。
uv run --no-sync python -m apps.cli file first.wav second.wav --model .runtime/models/whisper-tiny --language zh-CN
```

`--input-device` 是 devices 显示的 PortAudio 索引，`--device cpu|cuda` 是模型计算设备。默认采用系统输入设备，自动格式优先 16 kHz，再探测设备支持格式；显式索引或采样率无效时失败，不静默换设备。默认采集块为 1024 个源帧，缓冲预算为 5 秒；溢出和容量满明确报错。

file 和 stdin 默认 manual 且不启用实时。显式 `--mode auto` 使用自动分句，默认用内置 Silero v6，也可用 `--vad-model <path>` 换模型；`--realtime` 开启实时观察。mic 默认 auto 和 realtime，`--no-realtime` 关闭实时。默认单段容量 30 秒，`--max-audio-seconds` 可设为大于零且不超过 600 秒。

stdin 接收原始小端 PCM16，默认 16 kHz 单声道；float32、其他采样率或声道应显式声明，不接收 WAV 文件头。PowerShell 的文本管道可能转换二进制，向子进程写原字节时使用二进制流。以下示例只演示把已经持有的 PCM16 交给 CLI：

```python
import subprocess

completed = subprocess.run(
    ["python", "-m", "apps.cli", "stdin", "--model", "models/whisper-tiny", "--output", "jsonl"],
    input=pcm16_bytes,  # 调用方已有的原始音频字节。
    capture_output=True,
)
print(completed.returncode, completed.stdout.decode("utf-8"))
```

text 的 stdout 只包含每个最终正文一行，实时预览、状态和错误写 stderr。jsonl 的 stdout 是 UTF-8 逐行对象，保留完整核心事件、输入序号和来源；每行立即刷新。devices 的设备列表是该命令正文。标准流由进程拥有，CLI 不主动 close。

正常 EOF 或时长结束后，CLI 封口并排空最终结果。第一次 Ctrl+C 请求停止输入和排空，第二次取消剩余工作。退出码为成功 0、运行失败 1、参数无效 2、排空或清理未完成 4、用户中断 130；排空默认为 30 秒，关闭默认为 5 秒。坏管道不重试输出，空闲 stdin 不阻止已知事件故障的处理。

独立采集 SDK 不创建识别器：

```python
from realtime_asr import MicrophoneConfig, MicrophoneInput, list_input_devices

print(list_input_devices())
with MicrophoneInput(MicrophoneConfig()) as microphone:
    print(microphone.device, microphone.format)
    pcm = microphone.read_audio(timeout=1)
    print(len(pcm) if pcm is not None else 0)
```

将采集器的实际 `format` 用于 `SessionConfig`，再把原始块交给 `session.feed_audio()`。采集器不重采样，也不关闭外部会话。需要完整尾部时，先调用 `close()` 并检查 CloseResult，再读取尚未消费的已接收队列；pending 表示在途读取或清理仍未退出，不代表已释放。

具体契约见 [P5 Spec](docs/specs/microphone-and-cli.md)。`python -m evals.p5` 显式运行真实模型和本机设备验证，不作为产品运行入口。公开语料对照、真实设备采集和人工口述质量分别记录；当前实际验证及未验收事项见 [P5 Plan](docs/plans/p5-microphone-and-cli.md)。

## Python SDK

宿主可使用自己的 CPython 3.12 环境安装 wheel，不要求在项目目录运行：

```powershell
uv pip install "<wheel-path>"
```

默认安装包含 Whisper 后端、音频转换和 SoundFile/SoXR。最快的两种用法：

```python
from realtime_asr import transcribe_audio, transcribe_stream

# 转写一段已有音频，返回 Transcript
text = transcribe_audio("samples/speech.wav", model="tiny", language="zh").text

# 持续读取麦克风，每段产出一个 Transcript
for transcript in transcribe_stream(model="tiny", language="zh"):
    print(transcript.text)
```

`transcribe_stream` 默认使用库内置的 Silero v6 检测模型；要换模型时传 `vad_model=`。检测运行库来自 `vad` 附加依赖。需要复用模型或控制会话时，再用下面的类写法。

`model` 可以是模型名，也可以是已经准备好的目录；名字由所选引擎解释：

```python
from realtime_asr import Transcriber, ModelConfig

with Transcriber() as transcriber:                      # 默认 Whisper tiny
    print(transcriber.transcribe_file("samples/speech.wav", language="zh").text)

with Transcriber(ModelConfig(model="small.en")) as transcriber:
    print(transcriber.transcribe_file("samples/speech.wav").text)
```

模型名首次使用时由 faster-whisper 下载到 HuggingFace 缓存。`download_root` 指定缓存目录，`local_files_only=True` 禁止联网。模型名与引擎的对应关系见 [Whisper 引擎说明](docs/engines/whisper.md)。

已准备的目录按 `model-manifest.json` 校验，不信任目录名。固定权重（Nemotron）需要先安装：

```powershell
asr-install-sherpa-models --root D:/models/sherpa-onnx --model 560ms
```

```python
from realtime_asr import Transcriber, ModelConfig

with Transcriber(
    ModelConfig(
        model="D:/models/sherpa-onnx/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        engine="nemotron",
    )
) as transcriber:
    ...
```

宿主自己的模型实现 `BaseTranscriptionEngine` 后注入：`Transcriber(engine=MyEngine())`。SDK 只准备、调用和关闭该实例，不要求本地模型路径。

完整 PCM16 字节使用 `transcriber.transcribe(pcm, format=AudioFormat(), language="zh")`。完整浮点数组声明真实采样率与声道，例如 `AudioFormat(48000, 2, "float32")`。

有限音频块序列使用 `transcriber.transcribe_chunks(chunks, format=AudioFormat(), language="en")`。SDK 先读取和连续转换全部输入，再调用一次模型；这不是实时文本接口。输入块允许空块，迭代正常结束后刷新尾部。迭代失败不会返回前缀成功。

需要单独转换时，可使用 `AudioConverter(format)` 的 `feed`、`finish` 和 `close`。`feed` 返回完整标准块，`finish` 返回滤波余量与短尾块；`AudioFrame` 的位置按 16 kHz 有效采样计数。`close` 丢弃余量，不代替 `finish`。

同一 `Transcriber` 可顺序复用模型；并发准备、识别或关闭返回 `RUNTIME_BUSY`。错误通过 `ASRError.code`、`operation` 和 `message` 判断。关闭返回 `CloseResult`，重复关闭保留首次结论；`reference_drop` 不承诺显存回到加载前数值。

Whisper 的 CUDA 推理由 CTranslate2 在当前进程查找 CUDA 和 cuDNN。使用 `device="cuda"` 前，应先按显卡驱动安装匹配的 CUDA 12 和 cuDNN 9，并让它们的库目录出现在系统搜索路径中；库不下载、不校验、不固定这些运行库的版本。`compute_type` 默认 `"default"`，由 faster-whisper 按设备选择，也可显式传 `"float16"`、`"int8_float16"` 等值。CPU 配置不涉及这些。

完整接口与输入规则见 [P1 Spec](docs/specs/external-audio-and-sdk.md)。独立调用示例见 [examples/transcribe.py](examples/transcribe.py)。

```powershell
uv run --no-sync python examples/transcribe.py <audio-path> --model <model-path> --language zh
uv run --no-sync python -m evals.p1
```

后一命令显式运行真实模型验收，默认使用已准备的 `.runtime` 和 P0 公开语料。结果写入 `artifacts/p1/sdk-check/summary.json`；标准输入对照 P0 已保存探索结果，不形成正式 CER／WER。

## 连续会话 SDK

运行时显式准备共享模型，创建会话后按单元输入。`end_turn()` 固定完整音频并提交后台任务，返回后可以开始下一单元；它不表示推理已经完成。

```python
from realtime_asr import ModelConfig, SessionConfig
from realtime_asr.runtime.asr import ASRRuntime

with ASRRuntime(ModelConfig(model="models/whisper-tiny")) as runtime:
    session = runtime.create_session(SessionConfig(language="zh"))
    for pcm in (first_pcm, second_pcm):  # 示例：两个完整 PCM16 音频单元
        session.start()
        session.feed_audio(pcm)
        session.end_turn()
    for _ in range(4):
        event = session.next_event(timeout=30)
        if event is None:
            raise TimeoutError("等待事件超时")
        if event.kind == "final":
            print(event.turn, event.result.text)
    session.close()
```

正常任务依次交付 `final`、`completed`；失败只交付 `failed`，取消只交付 `cancelled`。实际程序应按这些终结类型消费，示例中的四次消费只演示两次正常成功。事件可以定位会话、代次、单元、任务和封口采样范围。

`cancel()` 只取消活动单元；取消已提交任务应传入 `TurnRef`。`reset()` 取消该会话旧工作并增加代次。`Session.close()` 不隐含结束，也不关闭共享模型。重置和关闭直接返回取消记录，并清除未消费的旧事件。

运行时关闭默认等待 5 秒。`pending` 表示仍有后台资源未退出；运行任务结束后才清理模型，宿主可再次关闭并观察终态。取消不承诺中断本地模型调用。音频、任务和结果容量分别受限，提交容量不足时保留 `sealed` 音频供重试；完整规则见 [P2 Spec](docs/specs/continuous-sessions.md)。

```powershell
uv run --no-sync python examples/continuous_session.py <first.wav> <second.wav> --model <model-path> --language en
uv run --no-sync python -m evals.p2
```

连续会话示例读取 16 kHz、单声道 PCM16 WAV，不需要 audio 依赖。真实模型验收位于 `evals/p2/`，结果写入 `artifacts/p2/session-check/summary.json`，不作为正式质量评分。

## 基本实时识别 SDK

运行时通过 `RealtimeConfig` 配置实时角色，会话通过 `SessionConfig(realtime=True)` 启用。关闭实时的会话保留普通最终事件流程。默认 `mode="snapshot"` 使用 Whisper；`mode="native"` 使用 Nemotron CPU。`model=None` 共享最终模型，显式 `model=ModelConfig(...)` 创建独立实时模型实例。

```python
from realtime_asr import RealtimeConfig, ModelConfig, SessionConfig
from realtime_asr.runtime.asr import ASRRuntime

with ASRRuntime(
    ModelConfig(model="models/whisper-tiny"),
    realtime=RealtimeConfig(),
) as runtime:
    session = runtime.create_session(SessionConfig(realtime=True, language="zh"))
    session.start()
    for chunk in pcm_chunks:  # 示例：调用方按音频来源节奏提供标准 PCM16
        session.feed_audio(chunk)
        while event := session.next_event():
            print(event.kind, event.source, event.result)
    session.end_turn()
    while True:
        event = session.next_event(timeout=60)
        if event is None:
            raise TimeoutError("等待识别事件超时")
        print(event.kind, event.source, event.result)
        if event.kind in ("completed", "failed", "cancelled"):
            break
    session.close()
```

实时文本是可变观察。`source` 区分 `snapshot`、`native` 与 `final`，`audio_samples` 说明实际覆盖范围。每单元只保留最新未消费实时观察，普通最终与完成事件保持顺序。`realtime_failed` 只终止该单元的实时路径，输入与普通最终识别仍可继续。

快照封口后停止更新；原生流封口时补送短尾和重采样余量，结束并关闭流后再执行完整普通最终。共享模型串行运行；独立模型可同时推进。实时单元在开始时预留配置的最大采样容量，取消后仍在执行的调用继续占用物理容量，直到实际返回。接口、计时和关闭规则见 [P3 Spec](docs/specs/basic-realtime.md)。

标准 WAV 的节奏输入示例和真实模型验收：

```powershell
uv run --no-sync python examples/realtime_session.py <audio-path> --model <model-path> --language en
uv run --no-sync python -m evals.p3
```

真实评估使用本地模型与已有公开语料，分别记录中文、英文连续单元和 30 秒重复音频的实时范围、时序、最终正文对照、RSS 与容量释放。评估结果位于 `artifacts/p3/`；这些记录不代替人工语料核实、正式 CER／WER 或吞吐量验收。

## 自动分句

安装 `vad` 可选依赖。自动会话默认使用内置 Silero v6；要换检测模型时显式传 `VadConfig(model_path=...)`。自动会话直接接收持续输入，不逐段调用 `start()`。一个输入代次使用同一个转换器，结束录音段不会重置重采样状态。

```python
from realtime_asr import ModelConfig, SessionConfig, VadConfig
from realtime_asr.runtime.asr import ASRRuntime

with ASRRuntime(ModelConfig(model=".runtime/models/whisper-tiny")) as runtime:
    session = runtime.create_auto_session(
        vad=VadConfig(model_path=".runtime/models/vad/silero-v6.onnx"),
        config=SessionConfig(language="zh-CN"),
    )
    for chunk in audio_chunks:
        session.feed_audio(chunk)
        while event := session.next_event():
            print(event.kind, event.boundary, event.result)
    session.end_input()
    while session.pending_segments:
        event = session.next_event(timeout=30)
        if event is None:
            raise TimeoutError("等待自动录音段超时")
        print(event.kind, event.boundary, event.result)
    session.close()
```

自动会话通过 WebRTC 与 Silero 确认开口，默认使用 WebRTC 判断句尾。默认保留 1 秒前置缓冲；保守裁剪与回溯均由 `SegmentationConfig` 显式启用。边界使用连续标准样本位置；开启回溯时暂留最新尾部，增加相应实时延迟，避免左右原生流重复接收音频。最终任务容量不足时保留待递交段，消费结果后继续递交；音频预算不足时整块拒绝，可消费后重送。

检测参数需要适合输入音量。本机英文公开录音的 RMS 约为 -49.9 dBFS，默认 `webrtc_mode=3` 漏检明显；验证中明确使用 `webrtc_mode=0`，没有自动放大源音频。此配置事实与首词、边界的人工质量验收不能混为一谈。检测模型身份、失败处理与控制契约见 [P4 Spec](docs/specs/automatic-segmentation.md)。

以下准备命令只复用已安装 faster-whisper 1.2.1 的固定检测模型资产；正式 SDK 只读取提供的路径，不依赖 faster-whisper 来定位检测模型：

```powershell
uv run --no-sync python -m evals.p4 --prepare-vad
uv run --no-sync python examples/automatic_session.py <audio-path> --model .runtime/models/whisper-tiny --vad-model .runtime/models/vad/silero-v6.onnx --language zh-CN
uv run --no-sync python -m evals.p4
```

真实评估代码位于 `evals/p4/`，记录保存到 `artifacts/p4/`。标准 PCM 示例只需 `whisper,vad`，不需要 `audio`；更换源格式时沿用 P1 的音频转换依赖与约束。模型文件需要另行准备，不包含在 wheel 中。

## 稳定文本与结束优化

这些策略默认关闭。`SessionConfig.stabilization` 为实时观察增加完整稳定前缀、待定尾部和本次稳定增量；最终事件提供一致性诊断，不改写最终正文。慢消费者应使用完整 `stable_text` 恢复显示，不能只拼接可能被最新槽替换的 `stable_delta`。

以下是自动会话配置示例。`runtime` 应已准备 Whisper 最终模型和快照实时角色；音频输入与独立事件消费循环沿用上一节。

```python
from realtime_asr import EndpointConfig, SessionConfig, StabilizationConfig, VadConfig

def open_optimized_session(runtime):
    return runtime.create_auto_session(
        config=SessionConfig(
            realtime=True,
            stabilization=StabilizationConfig(),
            word_timestamps=True,
        ),
        vad=VadConfig(model_path=".runtime/models/vad/silero-v6.onnx"),
        endpoint=EndpointConfig(
            early_transcription_silence=0.2,
            dynamic_silence=True,
            punctuation_split=True,
        ),
    )
```

`event.stabilization` 提供 `stable_text`、`unstable_text`、`stable_delta`、`display_text` 及接受／冲突诊断。`event.finalization` 提供最终前缀是否一致及差异原因。`LiveSession.optimization_status` 是本输入代次的诊断副本，不能通过修改返回值改变任务。

`RealtimeConfig(acoustic_boundaries=True)` 在 Whisper 快照中启用声学触发和两次补识别。Nemotron 原生流继续按新增样本解码，不使用声学快照策略。Whisper 单次 SDK 调用也可传入 `word_timestamps=True`；结果中的 `words` 使用标准样本范围。Nemotron 当前明确返回 `timestamp_status="unavailable"`，不能开启依赖词时间的句末切段。

动态等待按最近接受的实时正文选择句末、未知和续说等待，继续经过原来的门控。句末策略只支持句末标点，使用真实词时间固定左侧并保留右侧。提前结果只有在封口音频与快照完全相同时才复用；后续静音也会使版本不同。提前推理会增加成本，启用前应检查对应设备上的测量。

```powershell
uv run --no-sync python -m apps.cli mic --model .runtime/models/whisper-tiny --stable-text --word-timestamps --dynamic-silence --punctuation-split --duration 10 --output jsonl
```

`word_timestamps=True` 时完整结果包含词时间。会话策略通过 `SessionConfig` 和自动会话参数选择；声学策略由 `RealtimeConfig(acoustic_boundaries=True)` 开启。CLI 的 `--stable-text`、`--dynamic-silence`、`--punctuation-split`、`--acoustic-boundaries` 使用同一套配置。

## 模型准备

Whisper 模型按名字由 faster-whisper 下载到缓存，不需要准备步骤。需要固定版本、离线分发或清单校验时，可先用 `prepare_whisper` 或本地快照写入指定目录，再用目录加载。

Nemotron 固定权重使用安装器写入显式目录：

```powershell
uv run --no-sync asr-install-sherpa-models --root D:/models/sherpa-onnx
```

默认安装 `all`（560ms、1120ms、80ms 三个变体）；只装其中一个时加 `--model 560ms`。安装器校验压缩包大小和 SHA-256，压缩包先落 `.partial` 支持续传，在临时目录展开成功后原子改名；`--offline` 只校验已有模型。引擎启动不下载、不替换权重。

评估矩阵仍用 `python -m evals.p0 prepare-models` 准备到 `.runtime`；那是评估入口，不是 SDK 的使用步骤。

## GPU 环境

CPU 路径不需要额外步骤。GPU 推理所需的驱动、CUDA 和 cuDNN 由使用者安装，库不负责准备或校验：

1. 安装 NVIDIA 驱动。
2. 安装与驱动匹配的 CUDA 12 和 cuDNN 9，把 `bin` 目录加入系统搜索路径。
3. 用 `device="cuda"`（或 CLI 的 `--device cuda`）运行。

Nemotron 的 sherpa-onnx 后端同样由使用者选择：默认 `provider="cpu"`，需要 CUDA 时传 `engine_options={"provider": "cuda", "device": 0}`，并自行安装支持 CUDA 的 sherpa-onnx 运行库。库不检查显卡或驱动，失败时直接给出底层运行库的真实错误。

## 公开语料与听音核实

```powershell
uv run --locked --extra nemotron python -m evals.p0 prepare-corpus
uv run --locked --extra nemotron python -m evals.p0 validate-corpus --exploratory
```

公开语料来自 Google FLEURS 的中文和英文录音。原始录音、来源索引、格式转换记录和增广参数保存在 `data/corpus/p0/`，清单为 `manifest.json`。已有清单不会被准备入口重写。

14 个样本包括 12 个基础样本与 2 个尾块变体。当前正文使用来源标注，语音边界使用能量生成草稿；这些标注尚未经过人工听音核实。`annotation_checked=false` 必须保留到实际核实完成。来源性别标签不能代替准确说话人身份。

核实顺序与记录要求见 [语料核实指南](docs/p0-corpus-review.md)。所有语音及增广样本均需核实，不能只检查母录音后将增广样本自动标为通过。调整正文、边界或准备记录时，应使用新的 `corpus_version`；改变音频时，还应更新文件哈希和依赖它的增广样本。

## 运行与结果

未经核实的语料只允许显式探索：

```powershell
uv run --locked --extra nemotron python -m evals.p0 run --exploratory
```

核实全部标注后，运行正式验证：

```powershell
uv run --locked --extra nemotron python -m evals.p0 validate-corpus
uv run --locked --extra nemotron python -m evals.p0 run
```

默认声明并执行 M01—M10。可用 `--matrix M01,M03` 选择局部检查；未选择项仍记录为 `skipped`，局部成功不使阶段通过。

每次运行创建 `artifacts/p0/<run_id>/`，保留 `records.jsonl`、`summary.json` 和边界测试输出。原始记录包含环境、模型、配置、样本、操作、观察、试次及资源。不同参数采用不同配置标识；实际返回的空文本与失败分开保存。

报告列出 A01—A16 的证据归属和状态。可用 `python -m evals.p0 report <run-directory>` 从已保存记录重新生成摘要；此操作不重新推理，也不读取后来改动的当前语料正文。原始 JSONL 不被覆盖。

探索运行保存原始识别和测量，但不生成正式 CER／WER。依赖草稿语音边界的等待数值标为待核实。正式评分只使用规定的第 1 次结束结果；重复、自动语言、GPU 和其他路径独立分组。阶段只有 `passed` 或 `passed_with_limit` 时返回退出码 0；失败、取消、受阻返回 2。

按 Ctrl+C 请求取消。入口停止新工作；不能中断的原生调用返回后，只保留迟到诊断，再关闭自有资源。取消请求不表示原生计算已立即停止。

## 开发检查

```powershell
uv run --no-sync python -m unittest discover -s tests -t . -v
uv run --no-sync ruff check src evals tests examples apps
uv run --no-sync ruff format --check src evals tests examples apps
uv build
```

类和方法的中文注释说明职责与业务规则。模型外部边界替身用于失败及调度检查，不能作为真实模型的文本或性能证据。

Spec、实施 Plan 与状态入口分别为 [P0 Spec](docs/specs/runtime-and-corpus-baseline.md)、[P0 Plan](docs/plans/p0-runtime-and-corpus-baseline.md) 和 [STATUS](docs/STATUS.md)。

## 候选与同轮继续

宿主通过 `ASRRuntime(candidate=CandidateConfig())` 显式开启候选，默认复用主模型。独立模型使用 `CandidateConfig(model=...)` 或宿主注入的 `engine`。模型实例由运行时统一准备与关闭。会话的 `request_candidate(kind="tail"|"whole", hint=False)` 异步返回固定范围的请求，结果从 `next_event()` 消费。候选不会自动成为最终正文。

手动接收期间调用 `resume()`，保留完整音频历史并更新继续代次；旧代次预览不再发布。`SessionConfig(completion_mode="candidate")` 让候选角色执行完成任务，只复用同版本有效整轮候选。默认 `final` 策略保持原来的最终模型。

`retain_for_correction=True` 只保留最近完成单元，`request_correction(unit)` 的结果关联原最终任务，失败保留原正文。开始新单元、重置、取消对应单元和关闭会释放保留上下文。完整程序见 [候选示例](examples/candidates.py)。

本地启动器支持 `--candidates`、独立 `--candidate-model` 和 `--candidate-only`，以及 `--candidate-kind`、`--candidate-every`、`--candidate-hint`、`--completion-mode`、`--retain-for-correction` 和手动模式的 `--correct-after-final`。周期按输入音频时长计算；文本输出的候选写 stderr，最终正文写 stdout。

CLI 在手动采集时按 `--completion-mode` 和 `--retain-for-correction` 选择完成策略；同轮继续仅用于手动采集，修正成功更新原最终正文。候选计算期间仍继续递交麦克风音频。

## 可选唤醒与完整工作台

默认关闭唤醒。自动持续输入可以在宿主提供 `WakeConfig` 后启用；手动会话和 HTTP 整段保持直接识别。SDK 的 `create_auto_session(..., wake=...)` 使用本地 CPU Porcupine 或 OpenWakeWord，检测状态按输入代次隔离。激活宽限、正文等待、固定前置裁剪与继续窗口都按标准音频样本位置推进。

Porcupine 安装 `wake-porcupine` extra，设置 `PICOVOICE_ACCESS_KEY`，选择内置 keywords 或本地 .ppn 文件；非英语参数使用本地 .pv。密钥不写入命令参数。OpenWakeWord 安装 `wake-openwakeword` extra，显式提供关键词 .onnx、melspectrogram.onnx 和 embedding_model.onnx。模型准备时不隐式联网。项目 uv 配置只排除 openwakeword 0.6.0 的 TFLite 依赖，因为当前路线固定使用 ONNX。

本地 CLI 使用 `--wake`、`--wake-backend`、`--wake-keywords` 或 `--wake-models`、`--wake-feature-dir`，并可配置 `--wake-activation-delay`、`--wake-speech-timeout`、`--wake-trim-duration` 与 `--wake-followup-window`。默认继续窗口为零，独立唤醒词与正文之间有停顿时，应按实际使用配置有限窗口。

公开 `Session.current_audio()` 提供当前单元已经确定归属的标准音频副本及半开范围。自动模式不包含尚未确定归属的回溯尾部；观察不结束输入。可以通过 [录音保存示例](examples/save_audio.py)保存标准 WAV，或使用 [热键听写示例](examples/hotkey_dictation.py)在终端用 Space 开始／结束、Esc 取消并退出。热键不接管其他应用。

Porcupine 的正向实际检测仍需 AccessKey，人工口述、首词裁剪质量、终端热键与其他平台应分别验收。当前验证范围见 STATUS，不把受控检测当成人工设备验收。


## 模型工具

正式资源命令不加载识别模型，默认离线。显式准备与校验示例：

```powershell
uv run --no-sync python -m apps.cli models list
uv run --no-sync python -m apps.cli models prepare --kind whisper --path .runtime/models/whisper-tiny --download
uv run --no-sync python -m apps.cli models verify --path .runtime/models/whisper-tiny
```

模型工具支持 Whisper tiny、固定 Silero v6 和 OpenWakeWord Alexa ONNX 资源；Nemotron 固定权重统一使用 `asr-install-sherpa-models`。Porcupine 自定义文件和 AccessKey 由使用者提供。下载必须明确 --download；损坏或冲突文件不会被重写为成功缓存。

本次在四种新建 Windows 环境及项目外完成实际 CPU／GPU、CLI 和保存示例验证。支持仅限实测组合；正式语料评分、人工设备、Porcupine 正向及双实时收益没有被这些结果代替。
