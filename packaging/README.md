# 构建、模型与 CPU 服务部署

源码构建使用 Node 24、Python 3.12 和 uv。运行 wheel 不需要 Node，也不需要 evals 或开发目录。

## 准备与构建

在项目目录执行：

```powershell
uv sync --extra models --extra server --extra whisper --extra nemotron --extra audio --extra vad --extra wake-openwakeword --extra wake-porcupine
.venv/Scripts/python packaging/build.py --install-web-deps
```

构建器只清理项目内的 build 生成目录，先核对目标不是外部链接。它编译最新静态资源，构建 wheel，再比对全部生产包成员和字节，拒绝旧资源或遗漏。

wheel 位于 dist。项目外可以新建虚拟环境并安装需要的 extras，例如：

```powershell
uv venv --python 3.12 C:/asr-env
uv pip install --python C:/asr-env/Scripts/python.exe "C:/path/realtime_asr-0.1.0-py3-none-any.whl[server,whisper,audio,vad]"
```

基础 SDK 只需要 NumPy。远程客户端安装 client，读取文件再安装 audio；本地识别按后端安装 whisper 或 nemotron；设备输入另装 microphone。按用途安装，不复制其他项目的 site-packages。Linux／Python 3.12 的 OpenWakeWord ONNX 安装使用 `--excludes packaging/wake-onnx-excludes.txt`，排除该路线不使用的 TFLite 依赖；项目源码 uv 配置已有针对 0.6.0 的同等约束。

## 正式资源命令

下面的 uv run 使用当前已同步环境，不在每次调用时删除已选择 extras。

```powershell
uv run --no-sync realtime-asr models list
uv run --no-sync realtime-asr models prepare --kind whisper --path .runtime/models/whisper-tiny --download
uv run --no-sync realtime-asr models prepare --kind nemotron --variant 560ms --path .runtime/models/nemotron --download
uv run --no-sync realtime-asr models prepare --kind vad --path .runtime/models/vad --download
uv run --no-sync realtime-asr models prepare --kind openwakeword --path .runtime/models/wake --download
uv run --no-sync realtime-asr models verify --path .runtime/models/whisper-tiny
```

联网只在明确 --download 时发生。Whisper 可通过 --local-snapshot 复用固定修订；VAD／OpenWakeWord 可通过 --local-source 复制固定文件。缓存身份或内容不符时明确失败，不重写清单掩盖损坏，不覆盖冲突模型。Nemotron 1120ms 和 80ms 变体需要分别准备到不同目录。

Porcupine 内置词由可选包提供，自定义 .ppn／非英语 .pv 由操作员提供，并配置 PICOVOICE_ACCESS_KEY。它不使用 OpenWakeWord 的关键词文件。

Windows GPU 原生库单独准备：

```powershell
uv run --no-sync realtime-asr gpu-runtime prepare --root .runtime --download
uv run --no-sync realtime-asr gpu-runtime verify --path .runtime/cudnn/9.16.0.29
```

cuDNN 留在项目指定目录，CUDA 12.8 bin 可以显式复用系统目录。工具不修改系统驱动或 PATH。verify 只证明文件一致，实际 GPU 推理另行验证。当前 DLL 加载器只验证 Windows，本 CPU 镜像不提供 Linux GPU。

## CPU 镜像

```powershell
docker build -f packaging/Dockerfile -t realtime-asr:cpu .
```

镜像使用最新 wheel、Python 3.12、uv 和 CPU 后端，包含浏览器静态资源。运行用户为 asr，不包含麦克风或模型。模型与 TLS 文件从宿主只读挂载，启动不在线下载。

准备包含 REALTIME_ASR_TOKEN 的环境文件，以及 TLS 的 server.crt／server.key。健康探针默认验证挂载的 server.crt；证书应包含探针使用的 127.0.0.1，或通过 ASR_HEALTH_URL／ASR_HEALTH_CA 指定正确的主机及信任链。Token、私钥与录音不要放进构建上下文。

本机只发布回环端口：

```powershell
docker run -d --name realtime-asr --env-file C:/asr-secrets/runtime.env -p 127.0.0.1:8000:8000 --mount type=bind,src=C:/asr-models,dst=/models,readonly --mount type=bind,src=C:/asr-secrets/tls,dst=/run/asr-tls,readonly realtime-asr:cpu
```

默认模型为 /models/whisper-tiny，VAD 为 /models/vad/silero-v6.onnx，TLS 材料位于 /run/asr-tls。需要候选、唤醒或其他模型时，在镜像名后明确传入原有 serve 参数；覆盖 CMD 时同时保留 host、模型、TLS 等必要参数。

例如唤醒需要 `--wake --wake-models /models/wake/alexa_v0.1.onnx --wake-feature-dir /models/wake`；候选可加 --candidates。它们由宿主配置，客户端再申请已有能力。

浏览器访问实际 HTTPS 地址，在页面内填写 Token。程序使用 RemoteClient(url, token=..., verify=...)；测试自签证书使用对应 SSLContext，正式地址使用正确证书及信任链。/api/v1/ready 返回 200 后才接纳识别；/live 只表示进程存活。

正常停止：

```powershell
docker stop --time 30 realtime-asr
```

停止进入现有 Uvicorn 生命周期与资源关闭。部署记录检查实际退出及 OOM 状态，不把强杀当成正常关闭。支持组合及尚未完成的产品验收见项目 STATUS；安装、容器或浏览器模拟通过都不等于正式识别质量通过。
