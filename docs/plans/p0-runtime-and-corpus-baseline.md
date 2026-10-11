# 计划：P0 运行环境与识别基线

## 目标

交付本地验证入口，完成两种 CPU 后端的整段、快照及原生流式检查，保存可追溯的配置、语料、观察、评分和资源记录。GPU 对照按实际条件执行；阶段结论遵守 Spec，不以自动化测试代替真实模型验收。

## 依据

- [P0 Spec](../specs/runtime-and-corpus-baseline.md)：M01—M10、F01—F12、A01—A16。
- [架构技术栈](../ARCHITECTURE.md#技术栈)：环境、依赖、模型适配及 GPU 归属。
- [产品范围](../PRD.md#识别后端与验证环境)、[P0 规划](../ROADMAP.md#p0运行环境与语料基线)。
- [测试规范](../standards/testing-standard.md)、[文档规则](../standards/writing-rules.md)。

## 范围

按以下依赖执行；任务状态以对应证据为准，实际结果统一进入 [STATUS](../STATUS.md)。

| 任务 | 修改对象与结果 | 依赖及完成证据 | 状态 |
| --- | --- | --- | --- |
| T01 | 建立 `pyproject.toml`、`.python-version`、`uv.lock`、`src/realtime_asr/` 和测试入口；使用现有 uv Python 3.12.14 建立 `.venv`。 | 固定后端版本可安装，解释器实际路径及依赖版本有记录；不修改其他环境。 | 已完成 |
| T02 | 在 `core/audio.py` 实现标准 WAV 读取、数组与采样率校验；定义结果和可分类错误。 | T01；F03、F04 的正常输入与失败范围行为测试通过。 | 已完成 |
| T03 | 在 `runtime/models/` 实现显式准备、不可变修订、逐文件清单与离线校验。 | T01；F01、F02：完整复用、部分缺失、篡改和禁网加载检查有证据。 | 已完成 |
| T04 | 在 `core/engines/whisper.py` 接入固定批处理参数，消费全部分段，记录实际默认值和语言来源。 | T02、T03；分段消费行为测试及 W-CPU 真实整段转写。 | 已完成 |
| T05 | 在 `core/engines/nemotron.py` 与 `core/engines/stream.py` 实现共享识别器及独立流；语言、范围、结束、重置、缓存和释放按状态表执行。 | T02、T03；F04—F09 行为测试及 N-CPU 真实流式结束。 | 已完成 |
| T06 | 在项目级 `evals/p0/` 实现单调时钟、整段与流式试次、绝对回放和快照机会合并；取消与外部失败保留诊断。 | T04、T05；F10、F11 的控制时序测试，真实中间结果与独立完整结果。 | 已完成 |
| T07 | 在 `evals/p0/corpus/` 实现清单校验和显式增广，准备 12 个基础样本及 T01、T02；固定代表样本、来源及人工标注。 | T02；覆盖、哈希、区间、尾块和增广参数校验通过；人工听音证据独立保留。 | 已准备；人工听音待完成 |
| T08 | 在 `evals/p0/evaluation/` 实现规范化、编辑距离、分组评分、重复对照及完整性判断。 | T06、T07；A12 示例、空输出、非语音及主评分隔离测试通过。 | 实现与自动化通过；正式评分待核实语料 |
| T09 | 在 `evals/p0/diagnostics/` 实现环境信息、100 ms 资源观测、阶段与试次 JSONL；建立 `python -m evals.p0` 验证入口。 | T01、T06、T08；字段归属、缺失原因和原始观察可追溯。 | 已完成 |
| T10 | 显式准备项目 GPU 运行库并选择系统 CUDA 目录；记录文件与实际加载组合。 | T01、T03；W-GPU 初始化及推理结果，或明确的 blocked／failed 原因。 | 已完成；动态绑定版本及显存有记录限制 |
| T11 | 执行 CPU M01—M08、M10 及条件性 M09；预热、重复、重新加载、分块一致性和流隔离逐项保存。 | T03—T10；矩阵无静默缺项，真实文本、时间与资源记录可定位。 | 完整探索矩阵完成；正式矩阵待听音 |
| T12 | 核对 A01—A16、构建 wheel 并验证独立安装；补使用命令，同步 STATUS 和已有入口中的实际起点。 | T11；阶段报告与证据一致，未完成验收明确保留。 | 构建、安装及入口验证通过；正式验收受阻 |
| T13 | 分离 `core/`、`runtime/` 与项目级 `evals/p0/`，同步导入、测试分组和入口；限定正式包范围，建立本地 Git 管理。 | 用户确认的代码与评估分离要求；现有行为、独立安装、两种 CPU 后端调用和包内容检查通过。 | 已完成 |

识别核心放在 `src/realtime_asr/core/`，模型与原生运行库支持放在 `src/realtime_asr/runtime/`。P0 的语料、试次、测量、评分和验收放在项目级 `evals/p0/`，不进入 wheel；测试按 `core/`、`runtime/` 和 `evals/` 分组。每个模块围绕自身职责提供接口，不把模型下载、状态机、评分和命令解析堆在一个脚本。类与方法添加中文职责注释，业务条件和错误处理按逻辑分组。

模型放在 `.runtime/models/`，uv 缓存放在 `.runtime/uv-cache/`，语料放在 `data/corpus/p0/`，实测记录放在 `artifacts/p0/<run_id>/`。这些运行产物不进入普通 wheel；可复用清单及来源记录随相应产物保存。项目自有 GPU 运行库使用架构约定目录。

## 约束

- 只执行 P0，不开发麦克风、UI、网络服务或完整产品 SDK，不开展 Kroko、Qwen 或 FunASR。
- 不改动 Spec 的固定参数、文件校验值、矩阵或验收条件。实际不兼容应保留失败证据。
- 不修改系统驱动、全局 PATH、Anaconda 或其他项目环境，不从其他项目借用 cuDNN 作为部署依赖。
- 新增业务行为先确认测试因缺失行为失败，再实现并复核；配置、环境与安装使用适用的集成检查；替身只放在模型、时间、网络等外部边界。
- 人工标注未经听音核实，不写为 `annotation_checked=true`，不将探索试次计入正式评分。
- 下载或某后端受阻时，继续可独立完成的任务；无法取得必需真实证据时，阶段保持未完成。

## 验证

按 Spec 的矩阵和验收表执行，具体入口由本计划安排：

- `uv run --locked --extra whisper --extra nemotron python -m unittest discover -s tests -t . -v`：公开操作、状态、故障和评分行为。
- `uv run --locked --extra whisper --extra nemotron python -m evals.p0 environment`：实际解释器、硬件与依赖记录。
- `uv run --locked --extra whisper --extra nemotron python -m evals.p0 prepare-models`：显式模型准备及固定校验。
- `uv run --locked --extra whisper --extra nemotron python -m evals.p0 validate-corpus`：语料覆盖、标注、来源、格式与哈希。
- `uv run --locked --extra whisper --extra nemotron python -m evals.p0 run`：固定配置的真实矩阵、原始记录及汇总。
- `uv build`：打包；在独立临时环境中验证核心导入与后端调用，在项目源码中验证评估入口，不复用开发目录的隐含路径。

实施前已核对顺序：模型准备与语料准备可独立推进，正式测量依赖两者固定。人工听音是正式语料的必需条件，不能由模型输出补齐；GPU 失败按 Spec 形成限制，不阻止独立 CPU 工作。

完成条件为 Spec 的 CPU 核心证据齐全，M07、M09 按条件检查，A01—A16 有明确结论。代码完成、自动化通过和 P0 真实验收分别记录；其中任何一项未完成，不混写为整个阶段完成。

### 代码归属调整的验证结果

本次只调整包归属、导入和入口，不新增识别行为。使用已有行为测试和实际调用验证，没有为目录移动新增 TDD 用例。

36 项行为测试及 Ruff 检查通过。新的评估入口完成帮助、草稿语料校验和边界测试发现检查。Whisper 与 Nemotron 各执行一次 CPU 代表样本转写，文本与既有探索记录一致。

wheel 与源码包构建通过。wheel 只包含核心与运行支持；源码包保留评估和测试源码，不包含运行库、模型、语料或验证产物。独立环境安装 wheel 后，核心导入和音频检查通过，未安装评估及测试模块，未加载模型运行库。源码检查未发现正式包依赖 `evals/` 或 `tests/`。

证据：[行为测试](../../artifacts/p0/package-reorganization-tests.txt)、[包边界检查](../../artifacts/p0/package-reorganization.json)、[独立安装](../../artifacts/p0/package-reorganization-install.json)、[实际 CPU 调用与边界入口](../../artifacts/p0/package-reorganization-runtime.json)。本次未重跑完整矩阵或 GPU 推理；正式语料及 P0 验收的未完成状态保留。
