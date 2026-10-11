# realtime-asr

独立实时 ASR 项目。当前交付 P0 的本地运行和语料验证入口，包括 faster-whisper 整段识别、周期前缀快照，以及 Nemotron 原生流式识别。

当前入口用于开发验证。正式 SDK、麦克风、自动分句、服务、浏览器界面和唤醒尚未实现。产品范围见 [PRD](docs/PRD.md)，系统设计见 [Architecture](docs/ARCHITECTURE.md)，实际验收见 [STATUS](docs/STATUS.md)。

## 代码结构

核心代码位于 `src/realtime_asr/core/`。音频处理、识别结果、错误和后端适配器都在这里。`runtime/` 提供模型准备、完整性校验及原生运行库支持。

```text
src/realtime_asr/
├── core/                    # 识别核心
│   ├── audio.py             # 标准音频读取与校验
│   ├── results.py           # 识别结果
│   ├── errors.py            # 识别错误
│   └── engines/
│       ├── whisper.py       # Whisper 整段与快照调用
│       ├── nemotron.py      # Nemotron 模型与流创建
│       └── stream.py        # 原生流输入、解码、结束、重置与关闭
└── runtime/                 # 模型与运行库支持
    ├── models/              # 准备、来源、固定配置与文件校验
    ├── gpu.py               # GPU 运行库准备与选择
    ├── native_libraries.py  # 原生运行库加载
    └── stages.py            # 模型准备与加载阶段观测

evals/p0/                    # P0 评估，独立于正式安装包
├── __main__.py              # python -m evals.p0
├── corpus/                  # 公开语料、增广与标注校验
├── evaluation/              # CER / WER、指标与分组评分
├── diagnostics/             # 环境、资源观测与 JSONL
└── ...                      # 矩阵、试次、取消、隔离检查与报告

tests/
├── core/                    # 识别核心行为
├── runtime/                 # 模型和运行库边界
└── evals/                   # 评估流程、评分与报告行为
```

核心与运行支持不得依赖 `evals/` 或 `tests/`。评估代码调用识别实现，再补充语料、试次、测量和评分信息。`evals/p0/engines.py` 的 `EnginePool` 只负责评估中的模型复用和预热，不作为产品运行时。

当前尚未实现正式 SDK、产品运行时及会话管理。后续按已确认架构建设，共用识别核心；不会把评估矩阵作为产品识别流程。

## 环境与安装

使用 uv 和 64 位 CPython 3.12。开发解释器由 `.python-version` 固定为已安装的 3.12.14，Python 依赖由 `uv.lock` 锁定。以下命令在项目根目录执行：

```powershell
uv sync --locked --extra whisper --extra nemotron
uv run --locked --extra whisper --extra nemotron python -m evals.p0 environment
```

uv 建立根目录 `.venv`，不向 Anaconda 或其他项目环境安装依赖。`uv run` 也应带上所需 extras；省略 extras 会按较小的依赖集合重新同步环境。

基础包只安装 NumPy，后端依赖按 extras 安装。P0 评估通过项目根目录的 `evals/` 运行，使用开发依赖组中的 SciPy 和 psutil，M10 使用 `tests/`。wheel 只包含 `src/realtime_asr/` 中的识别核心与运行支持，不包含评估工具、测试或语料。源码分发包保留评估和测试源码，供开发验证使用。

## 模型准备

模型文件位于 `.runtime/models/`，与虚拟环境分开。准备操作保存来源、不可变修订、逐文件大小和 SHA-256。已有清单只能校验复用，不能用新清单覆盖文件篡改。

```powershell
uv run --locked --extra whisper --extra nemotron python -m evals.p0 prepare-models --allow-download
```

Whisper 使用 `Systran/faster-whisper-tiny`。首次下载解析具体仓库修订后保存，后续加载只读取本地文件。也可通过 `--local-whisper-snapshot <path>` 显式复制已有的不可变快照；复制模型文件不复用该快照所在环境的 Python 包。

Nemotron 使用 Spec 固定的 Sherpa 3.5 INT8 560 ms 发布包。准备操作先校验发布包，再展开并核对固定模型文件。

没有 `--allow-download` 时只校验已有模型。运行矩阵不调用准备或下载入口；缺少模型时保存明确失败。

## GPU 准备

CPU 路径不需要执行此步骤。Windows GPU 验证显式使用系统 CUDA 12.8 的 `bin` 目录，以及项目 `.runtime/cudnn/9.16.0.29/bin`：

```powershell
uv run --locked --extra whisper --extra nemotron python -m evals.p0 prepare-gpu
```

准备入口获取 NVIDIA cuDNN 包，校验发布哈希并保存原生文件清单。项目进程持有所选 DLL 和目录句柄，不修改系统 PATH，也不使用其他项目虚拟环境内的 cuDNN。实际已加载 DLL 路径进入验证记录；同名库同时加载时，不仅凭路径推定实际绑定版本。详细归属见 [GPU 运行库](docs/ARCHITECTURE.md#gpu-运行库的位置与复用边界)。

## 公开语料与听音核实

```powershell
uv run --locked --extra whisper --extra nemotron python -m evals.p0 prepare-corpus
uv run --locked --extra whisper --extra nemotron python -m evals.p0 validate-corpus --exploratory
```

公开语料来自 Google FLEURS 的中文和英文录音。原始录音、来源索引、格式转换记录和增广参数保存在 `data/corpus/p0/`，清单为 `manifest.json`。已有清单不会被准备入口重写。

14 个样本包括 12 个基础样本与 2 个尾块变体。当前正文使用来源标注，语音边界使用能量生成草稿；这些标注尚未经过人工听音核实。`annotation_checked=false` 必须保留到实际核实完成。来源性别标签不能代替准确说话人身份。

核实顺序与记录要求见 [语料核实指南](docs/p0-corpus-review.md)。所有语音及增广样本均需核实，不能只检查母录音后将增广样本自动标为通过。调整正文、边界或准备记录时，应使用新的 `corpus_version`；改变音频时，还应更新文件哈希和依赖它的增广样本。

## 运行与结果

未经核实的语料只允许显式探索：

```powershell
uv run --locked --extra whisper --extra nemotron python -m evals.p0 run --exploratory
```

核实全部标注后，运行正式验证：

```powershell
uv run --locked --extra whisper --extra nemotron python -m evals.p0 validate-corpus
uv run --locked --extra whisper --extra nemotron python -m evals.p0 run
```

默认声明并执行 M01—M10。可用 `--matrix M01,M03` 选择局部检查；未选择项仍记录为 `skipped`，局部成功不使阶段通过。

每次运行创建 `artifacts/p0/<run_id>/`，保留 `records.jsonl`、`summary.json` 和边界测试输出。原始记录包含环境、模型、配置、样本、操作、观察、试次及资源。不同参数采用不同配置标识；实际返回的空文本与失败分开保存。

报告列出 A01—A16 的证据归属和状态。可用 `python -m evals.p0 report <run-directory>` 从已保存记录重新生成摘要；此操作不重新推理，也不读取后来改动的当前语料正文。原始 JSONL 不被覆盖。

探索运行保存原始识别和测量，但不生成正式 CER／WER。依赖草稿语音边界的等待数值标为待核实。正式评分只使用规定的第 1 次结束结果；重复、自动语言、GPU 和其他路径独立分组。阶段只有 `passed` 或 `passed_with_limit` 时返回退出码 0；失败、取消、受阻返回 2。

按 Ctrl+C 请求取消。入口停止新工作；不能中断的原生调用返回后，只保留迟到诊断，再关闭自有资源。取消请求不表示原生计算已立即停止。

## 开发检查

```powershell
uv run --locked --extra whisper --extra nemotron python -m unittest discover -s tests -t . -v
uv run --locked --extra whisper --extra nemotron ruff check src evals tests
uv run --locked --extra whisper --extra nemotron ruff format --check src evals tests
uv build
```

类和方法的中文注释说明职责与业务规则。模型外部边界替身用于失败及调度检查，不能作为真实模型的文本或性能证据。

Spec、实施 Plan 与状态入口分别为 [P0 Spec](docs/specs/runtime-and-corpus-baseline.md)、[P0 Plan](docs/plans/p0-runtime-and-corpus-baseline.md) 和 [STATUS](docs/STATUS.md)。
