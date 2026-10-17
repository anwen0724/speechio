"""从保存的机器记录重建报告，不读取变更后的当前语料或重新推理。"""

import json
from pathlib import Path
from types import SimpleNamespace

from evals.p0.report import build_report
from realtime_asr.errors import ASRError


def rebuild_report(folder):
    """评分使用运行时保存的正文和标注状态，旧版本不能借新标注通过。"""
    folder = Path(folder).resolve()
    samples, trials, checks, configurations = {}, [], [], {}
    environment, resources, native_runtime, corpus = {}, {}, [], None
    run_id = None
    path = folder / "records.jsonl"
    with path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if run_id is not None and item["run_id"] != run_id:
                raise ASRError("RECORD_INCOMPLETE", "rebuild_report", "记录混入其他运行身份")
            run_id = item["run_id"]
            match item["record_type"]:
                case "sample":
                    samples[item["sample_id"]] = item
                case "trial_result":
                    trials.append(item)
                case "configuration":
                    configurations[item["config_id"]] = item
                case "boundary_check":
                    checks.append(item)
                case "environment":
                    environment = item
                case "corpus":
                    corpus = item
                case "native_runtime":
                    native_runtime.append(item)
                case "resource_sample":
                    key = f"{item.get('backend', item.get('config_id'))}:{item['phase']}"
                    group = resources.setdefault(
                        key,
                        {
                            "samples": 0,
                            "observed_peak_rss_bytes": None,
                            "observed_peak_gpu_bytes": None,
                            "gpu_memory_scopes": [],
                            "unavailable_reasons": [],
                            "peak_scope": "采样观察峰值",
                        },
                    )
                    group["samples"] += 1
                    for name, field in [
                        ("observed_peak_rss_bytes", "rss_bytes"),
                        ("observed_peak_gpu_bytes", "gpu_memory_bytes"),
                    ]:
                        if item.get(field) is not None:
                            group[name] = max(group[name] or 0, item[field])
                    scope = item.get("gpu_memory_scope")
                    if scope is not None and scope not in group["gpu_memory_scopes"]:
                        group["gpu_memory_scopes"].append(scope)
                    reason = item.get("unavailable_reason")
                    if reason and reason not in group["unavailable_reasons"]:
                        group["unavailable_reasons"].append(reason)
    if not run_id or not samples or not trials or corpus is None:
        raise ASRError("RECORD_INCOMPLETE", "rebuild_report", "语料快照、样本或试次记录缺失")
    storage = SimpleNamespace(
        trials=trials,
        samples=samples,
        configurations=configurations,
        corpus=corpus,
        writer=SimpleNamespace(run_id=run_id, path=path),
        environment=environment,
        resource_summary=resources,
        native_runtime=native_runtime,
    )
    exploratory = any(item.get("exploratory") for item in trials)
    report = build_report(storage, checks, exploratory=exploratory)
    (folder / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
