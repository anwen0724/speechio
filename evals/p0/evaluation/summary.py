"""只对规定的结束结果评分；阶段结论保留缺项与能力限制。"""

from evals.p0.evaluation.scoring import score

FINAL_KINDS = {"whole_final", "snapshot_final", "stream_final"}


def grouped_scores(trials, samples):
    """主评分、重复、自动语言及各设备路径分别归组。"""
    sample_map = {item["sample_id"]: item for item in samples}
    groups = {}
    for trial in trials:
        if trial["evaluation_role"] == "warmup":
            continue
        if trial.get("output_kind") not in FINAL_KINDS:
            if trial.get("status") == "ok":
                continue
        if trial["evaluation_role"] == "primary" and trial.get("trial_index") != 1:
            raise ValueError("主评分只能使用首次正式试次")
        sample = sample_map[trial["sample_id"]]
        key = (
            trial["config_id"],
            trial["input_path"],
            trial["language_mode"],
            trial["corpus_version"],
            trial["evaluation_role"],
            sample["language"],
        )
        if key not in groups:
            groups[key] = {
                "config_id": key[0],
                "input_path": key[1],
                "language_mode": key[2],
                "corpus_version": key[3],
                "evaluation_role": key[4],
                "language": key[5],
                "N": 0,
                "E": 0,
                "expected_trials": 0,
                "successful_trials": 0,
                "failed_trials": 0,
                "cancelled_trials": 0,
                "missing_trials": 0,
                "non_speech_nonempty_trials": 0,
                "scores": [],
            }
        group = groups[key]
        group["expected_trials"] += 1
        if trial["status"] != "ok" or trial.get("result") is None:
            counter = {"failed": "failed_trials", "cancelled": "cancelled_trials"}.get(
                trial["status"], "missing_trials"
            )
            group[counter] += 1
            continue
        group["successful_trials"] += 1
        scored = score(sample["reference_text"], trial["result"]["text"], sample["language"])
        group["scores"].append(
            {"sample_id": sample["sample_id"], "trial_id": trial.get("trial_id"), **scored}
        )
        group["N"] += scored["N"]
        group["E"] += scored["E"]
        if sample["language"] == "none" and trial["result"]["text"].strip():
            group["non_speech_nonempty_trials"] += 1
    for group in groups.values():
        group["error_rate"] = group["E"] / group["N"] if group["N"] else None
        count = group["successful_trials"]
        group["non_speech_error_rate"] = (
            group["non_speech_nonempty_trials"] / count
            if count and group["language"] == "none"
            else None
        )
    return list(groups.values())
