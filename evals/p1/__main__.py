"""显式运行真实 SDK 验收，保存输入、P0 对照及实际关闭证据。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from realtime_asr import ASRError, AudioConverter, AudioFormat, Recognizer, RecognizerConfig
from realtime_asr.core.runtime.native.gpu import CUDNN_VERSION


def saved_baselines(path):
    """按保存配置和实际语言定位 P0 结果，不使用语料草稿正文评分。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    configs = {r["config_id"]: r for r in rows if r["record_type"] == "configuration"}
    baselines = {}
    for row in rows:
        if row.get("output_kind") not in ("whole_final", "stream_final"):
            continue
        config = configs[row["config_id"]]
        matrix = {"whisper": "M01", "whisper-gpu": "M09", "nemotron": "M03"}.get(config["backend"])
        if row.get("matrix_item") != matrix:
            continue
        if config["language_mode"] != "fixed" or row.get("status") != "ok":
            continue
        key = (config["backend"], row["sample_id"], row["requested_language"])
        baselines[key] = row["result"]["text"]
    return baselines


def check_backend(name, root, corpus, manifest, baselines, output):
    """一次准备复用模型，输入入口和原生流按每次调用独立执行。"""
    backend = "nemotron" if name == "nemotron" else "whisper"
    gpu = name == "whisper-gpu"
    config = RecognizerConfig(
        model=root / "models" / ("nemotron" if backend == "nemotron" else "whisper-tiny"),
        backend=backend,
        device="cuda" if gpu else "cpu",
        cudnn_folder=root / "cudnn" / CUDNN_VERSION if gpu else None,
    )
    recognizer = Recognizer(config)
    record = {
        "backend": name,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(recognizer.config).items()
        },
        "checks": [],
        "status": "running",
    }
    try:
        recognizer.prepare()
        model = json.loads(
            (recognizer.config.model / "model-manifest.json").read_text(encoding="utf-8")
        )
        record["model_identity"] = {
            k: model[k] for k in ("model_record_id", "model_id", "revision", "source")
        }
        for category in ("C01", "C07"):
            sample_id = manifest["representative_sample_id"][category]
            sample = next(s for s in manifest["samples"] if s["sample_id"] == sample_id)
            path = corpus / sample["audio_path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != sample["audio_sha256"]:
                raise ValueError(f"验收输入身份不符：{sample_id}")
            source, rate = sf.read(path, dtype="int16")
            if rate != 16000:
                raise ValueError("P0 对照输入应为标准采样率")
            language = sample["language"]
            requested = (
                {"zh": "zh-CN", "en": "en-US"}[language] if backend == "nemotron" else language
            )
            expected = baselines[(name, sample_id, requested)]
            pcm = source.astype("<i2").tobytes()
            entries = {
                "array": lambda: recognizer.transcribe(
                    source, format=AudioFormat(), language=language
                ),
                "chunks": lambda: recognizer.transcribe_chunks(
                    (pcm[start : start + 1001] for start in range(0, len(pcm), 1001)),
                    format=AudioFormat(),
                    language=language,
                ),
                "file": lambda: recognizer.transcribe_file(path, language=language),
            }
            for entry, call in entries.items():
                started = time.perf_counter()
                result = call()
                record["checks"].append(
                    {
                        "sample_id": sample_id,
                        "audio_sha256": digest,
                        "entry": entry,
                        "samples": len(source),
                        "seconds": time.perf_counter() - started,
                        "result": asdict(result),
                        "same_p0_text": result.text == expected,
                    }
                )
            # 真实语音经独立上采样生成 48 kHz 双声道文件，不改写原始语料。
            if category == "C01":
                upsampled = soxr.resample(source.astype(np.float32), 16000, 48000, quality="HQ")
                stereo = (
                    np.column_stack((upsampled, upsampled)).clip(-32768, 32767).astype(np.int16)
                )
                generated = output / f"{name}-48000-stereo.wav"
                sf.write(generated, stereo, 48000, subtype="PCM_16")
                converter = AudioConverter(AudioFormat(48000, 2))
                try:
                    frames = converter.feed(stereo) + converter.finish()
                    count = sum(f.end_sample - f.start_sample for f in frames)
                finally:
                    converter.close()
                result = recognizer.transcribe_file(generated, language=language)
                record["checks"].append(
                    {
                        "sample_id": sample_id,
                        "entry": "stereo-file",
                        "source_samples": len(stereo),
                        "standard_samples": count,
                        "result": asdict(result),
                        "not_p0_identity": True,
                    }
                )
        same = all(c.get("same_p0_text", True) for c in record["checks"])
        record["status"] = "ok" if same else "failed"
    except Exception as exc:
        record["status"] = (
            "blocked"
            if isinstance(exc, ASRError) and exc.code in ("DEPENDENCY_MISSING", "MODEL_INCOMPLETE")
            else "failed"
        )
        record["error"] = {
            "code": getattr(exc, "code", type(exc).__name__),
            "operation": getattr(exc, "operation", "eval"),
            "message": str(exc),
        }
    finally:
        record["close"] = asdict(recognizer.close())
        if record["close"]["status"] != "ok":
            record["status"] = "failed"
    return record


def main():
    """真实验收需用户显式执行此命令，不由单元测试自动加载模型。"""
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="P1 SDK 真实验收")
    parser.add_argument("--runtime", type=Path, default=project / ".runtime")
    parser.add_argument("--corpus", type=Path, default=project / "data/corpus/p0")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=project / "artifacts/p0/20261011T182121Z-34b49cd6/records.jsonl",
    )
    parser.add_argument("--output", type=Path, default=project / "artifacts/p1/sdk-check")
    parser.add_argument(
        "--backend", action="append", choices=("whisper", "nemotron", "whisper-gpu")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    baselines = saved_baselines(args.baseline)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "soundfile", "soxr", "faster-whisper", "sherpa-onnx")
        },
        "libsndfile": sf.__libsndfile_version__,
        "libsndfile_package_path": sf._full_path,
        "p0_formal_acceptance": "pending_human_review",
        "backends": [],
    }
    for backend in args.backend or ("whisper", "nemotron", "whisper-gpu"):
        record = check_backend(backend, args.runtime, args.corpus, manifest, baselines, args.output)
        report["backends"].append(record)
        (args.output / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{backend}: {record['status']}, {len(record['checks'])} checks", flush=True)
    return 0 if all(r["status"] == "ok" for r in report["backends"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
