"""从实际观察生成可比较指标，核对最终模型输入摘要与左右连续性。"""

import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(folder):
    rows = json.loads((folder / "matrix.json").read_text(encoding="utf-8"))
    comparisons = []
    for row in rows:
        finals = [event for event in row["events"] if event["kind"] == "final"]
        waits = [
            (event["received_at"] - event["submitted_at_monotonic"]) * 1000
            for event in finals
            if event["source"] == "final"
        ]
        calls = row["model_calls"]
        if calls is not None:
            inputs = {call["digest"] for call in calls if call["role"] == "final"}
            if any(audio["digest"] not in inputs for audio in row["ranges"]):
                raise AssertionError("最终范围未对应实际最终模型输入")
        if row["name"] == "punctuation" and row.get("punctuation_triggered", True):
            if row["optimization"].get("punctuation_splits", 0) == 0:
                raise AssertionError("真实长句未触发句末切段")
            for left, right in zip(row["ranges"], row["ranges"][1:]):
                if left["end"] != right["start"]:
                    raise AssertionError("真实句末左右范围不连续")
        comparisons.append(
            {
                "backend": row["backend"],
                "language": row["language"],
                "strategy": row["name"],
                "units": len(finals),
                "input_samples": row["input_samples"],
                "first_nonempty_realtime_ms": round(row["first_nonempty_realtime_ms"], 2),
                "sealed_task_to_final_mean_ms": round(mean(waits), 2) if waits else None,
                "source_eof_to_last_final_ms": round(row["finish_to_last_final_ms"], 2),
                "realtime_observations": sum(row["trigger_counts"].values()),
                "trigger_counts": row["trigger_counts"],
                "stable_observations": row["stable_observations"],
                "optimization": row["optimization"],
                "whisper_calls": len(calls) if calls is not None else None,
                "whisper_execution_total_ms": round(sum(call["execution_ms"] for call in calls), 2)
                if calls is not None
                else None,
                "final_matches_stable": [
                    event["finalization"]["agrees_with_stable_prefix"] for event in finals
                ],
                "punctuation_triggered": row.get("punctuation_triggered"),
            }
        )
    report = {
        "status": "passed",
        "cases": len(comparisons),
        "comparisons": comparisons,
        "measurement_notes": [
            "封口任务到交付包含排队、模型和消费；正式语音结束点尚未人工标注。",
            "自动最终可能先于源 EOF 到达，EOF 等待为零不能证明无端点等待。",
            "Whisper 计数完整执行真实外部模型，包括辅助任务；Nemotron 不伪造完整模型成本。",
            "峰值内存和显存未采样；物理任务容量由行为测试验证。",
            "此文件验证实际模型输入和句末左右连续，不代替人工误切与识别质量评分。",
        ],
    }
    (folder / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="核对并汇总 P7 已运行的真实结果")
    parser.add_argument("--folder", type=Path, default=Path("artifacts/p7/real-check"))
    report = summarize(parser.parse_args().folder)
    for row in report["comparisons"]:
        if row["strategy"] in ("vad-baseline", "early", "dynamic"):
            print(
                row["backend"],
                row["language"],
                row["strategy"],
                row["sealed_task_to_final_mean_ms"],
                row["whisper_calls"],
            )


if __name__ == "__main__":
    main()
