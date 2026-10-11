"""M10 本地边界、实际模型状态及矩阵一致性检查。"""

import io
import os
import socket
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from evals.p0.matrix import declare_matrix
from realtime_asr.core.errors import ASRError, error_record


def require(condition, message):
    """验收条件失败保持明确异常，不用 Python 可关闭的 assert。"""
    if not condition:
        raise ASRError("RECORD_INCOMPLETE", "boundary_check", message)


def run_tests(patterns, folder):
    """执行开发边界测试并保存实际输出；替身不提供质量基线。"""
    test_root = Path(__file__).resolve().parents[2] / "tests"
    require(test_root.is_dir(), "M10 自动化边界测试需要项目源码目录")
    output = io.StringIO()
    count = 0
    filename = "boundary-" + patterns[0].removesuffix(".py") + ".txt"
    temporary_root = folder / ".boundary-tmp"
    temporary_root.mkdir(exist_ok=True)
    with patch.object(tempfile, "tempdir", str(temporary_root)):
        for pattern in patterns:
            suite = unittest.TestLoader().discover(
                str(test_root), pattern=pattern, top_level_dir=str(test_root.parent)
            )
            result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
            count += result.testsRun
            (folder / filename).write_text(output.getvalue(), encoding="utf-8")
            require(result.wasSuccessful(), f"{pattern} 自动化边界测试失败")
    require(count > 0, "没有发现边界测试")
    return {
        "tests_run": count,
        "test_output": filename,
        "evidence_scope": "本地行为检查，不提供真实识别效果证据",
    }


def compare_chunks(storage):
    """最终文本逐字比较，不做规范化、去重或噪声过滤。"""
    comparisons = []
    variants = [item for item in storage.trials if item["matrix_item"] == "M05"]
    require(variants, "缺少 M05 分块变体")
    for variant in variants:
        candidates = [
            item
            for item in storage.trials
            if item["sample_id"] == variant["sample_id"]
            and item["matrix_item"] in ("M03", "M05")
            and item["status"] == "ok"
            and item.get("scenario") is None
            and item.get("trial_index") == 1
            and storage.configurations[item["config_id"]].get("chunk_samples") == 512
        ]
        require(
            variant["status"] == "ok" and candidates, f"{variant['sample_id']} 缺少成功分块对照"
        )
        base = candidates[0]
        require(
            variant["accepted_range"] == [0, storage.samples[variant["sample_id"]]["sample_count"]],
            "尾块或输入范围遗漏",
        )
        comparisons.append(
            {
                "trial_id": variant["trial_id"],
                "baseline_trial_id": base["trial_id"],
                "same_text": variant["result"]["text"] == base["result"]["text"],
            }
        )
    require(all(item["same_text"] for item in comparisons), f"分块最终文本不一致：{comparisons}")
    return {"comparisons": comparisons}


def compare_isolation(storage):
    """每个结果只与对应独立语言配置比，不把跨语言提示差异记作串流。"""
    comparisons = []
    isolated = [item for item in storage.trials if item["matrix_item"] == "M06"]
    require(len(isolated) == 4, "缺少重置或交错输入项")
    for item in isolated:
        expected_matrix = "M07" if item["scenario"] == "reset" else "M03"
        baseline = [
            other
            for other in storage.trials
            if other["matrix_item"] == expected_matrix
            and other["sample_id"] == item["sample_id"]
            and other["input_path"] == "stream"
            and other["status"] == "ok"
            and other.get("trial_index") == 1
        ]
        require(item["status"] == "ok" and baseline, "隔离检查缺少实际结果或独立对照")
        check = item.get("reset_check")
        if check:
            require(
                check["old_stream_id"] != check["new_stream_id"]
                and check["accepted_samples"] == 0
                and check["cache_text"] == "",
                "重置未形成独立的新流",
            )
        comparisons.append(
            {
                "trial_id": item["trial_id"],
                "baseline_trial_id": baseline[0]["trial_id"],
                "same_text": item["result"]["text"] == baseline[0]["result"]["text"],
            }
        )
    require(all(item["same_text"] for item in comparisons), f"隔离结束文本不一致：{comparisons}")
    return {"comparisons": comparisons}


def real_stream_states(engine, kind):
    """使用实际 Nemotron 流检查零输入、重复结束及关闭状态。"""
    stream = engine.create_streaming_session("zh-CN")
    try:
        if kind == "F05":
            empty = stream.finish()
            require(stream.finish() == empty, "零输入重复结束改变结果")
            stream.reset()
            stream.accept_audio(np.zeros(1600, np.float32))
            first = stream.finish()
            require(
                stream.finish() == first and stream.accepted_samples == 1600,
                "重复结束改变范围或结果",
            )
            return {"zero_input_result": asdict(empty), "result": asdict(first)}
        if kind == "F06":
            stream.accept_audio(np.zeros(160, np.float32))
            final = stream.finish()
            operations = [
                ("finished", lambda: stream.accept_audio([]), "STREAM_FINISHED"),
                ("closed", lambda: stream.accept_audio([]), "STREAM_CLOSED"),
                ("closed", stream.reset, "STREAM_CLOSED"),
            ]
            for state, operation, code in operations:
                if state == "closed":
                    stream.close()
                try:
                    operation()
                except ASRError as exc:
                    require(exc.code == code, "流状态错误分类不符")
                else:
                    require(False, "非法流状态调用没有失败")
                require(
                    stream.accepted_samples == 160 and stream.get_result() == final,
                    "状态错误改变范围或缓存",
                )
            return {"result": asdict(final)}
        stream.accept_audio(np.zeros(160, np.float32))
        stream.decode()
        before = stream.get_result()
        stream.close()
        cleanup = dict(stream.cleanup)
        stream.close()
        stream.decode()
        require(
            stream.get_result() == before
            and not stream.has_final_result
            and stream.cleanup == cleanup,
            "直接关闭伪造结束或重复关闭改写清理结论",
        )
        return {"cached_result": asdict(before), "cleanup": cleanup}
    finally:
        stream.close()


def run_boundaries(storage, pool, audio, cancel, selected):
    """预期错误按边界验收；缺少真实矩阵对照则保留受阻原因。"""
    checks = []
    task_map = {
        item.backend: item
        for item in declare_matrix(storage.corpus)
        if item.matrix_item in ("M01", "M03")
        and item.sample_id == storage.corpus["representative_sample_id"]["C01"]
        and item.trial_index == 1
    }

    def offline_reload():
        records = []

        def deny_network(*args, **kwargs):
            raise RuntimeError("禁网验证：不允许模型获取网络")

        with (
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}),
            patch.object(socket.socket, "connect", deny_network),
            patch.object(socket, "create_connection", deny_network),
        ):
            for backend in ("whisper", "nemotron"):
                record = pool.reload(task_map[backend], exploratory=True, cancel=cancel)
                require(
                    record is not None and record["status"] == "ok", f"{backend} 禁网重新加载失败"
                )
                records.append(record["trial_id"])
        return {
            "reload_trial_ids": records,
            "network_policy": "HF_HUB_OFFLINE=1; socket connect/create_connection 禁止",
        }

    actions = {
        "F01": lambda: run_tests(["test_models.py"], storage.writer.folder),
        "F02": offline_reload,
        "F03": lambda: run_tests(["test_audio.py"], storage.writer.folder),
        "F04": lambda: run_tests(["test_stream.py"], storage.writer.folder),
        "F05": lambda: real_stream_states(pool.engines["nemotron"], "F05"),
        "F06": lambda: real_stream_states(pool.engines["nemotron"], "F06"),
        "F07": lambda: real_stream_states(pool.engines["nemotron"], "F07"),
        "F08": lambda: compare_chunks(storage),
        "F09": lambda: compare_isolation(storage),
        "F10": lambda: run_tests(
            ["test_snapshots.py", "test_snapshot_run.py"], storage.writer.folder
        ),
        "F11": lambda: run_tests(["test_trials.py", "test_snapshot_run.py"], storage.writer.folder),
        "F12": lambda: run_tests(
            ["test_corpus.py", "test_gpu.py", "test_trials.py"], storage.writer.folder
        ),
    }
    for identifier, action in actions.items():
        config_id = f"boundary-{identifier}"
        sample_id = f"boundary-{identifier}-cases"
        storage.writer.write(
            "boundary_configuration",
            config_id=config_id,
            matrix_item="M10",
            operation=identifier,
            purpose="API 状态、输入或故障契约；不参与质量评分",
        )
        storage.writer.write(
            "boundary_sample",
            sample_id=sample_id,
            source="本地边界场景及实际矩阵对照",
            corpus_version="p0-boundary-v1",
        )
        item = {
            "check_id": identifier,
            "matrix_item": "M10",
            "config_id": config_id,
            "sample_id": sample_id,
            "trial_id": uuid4().hex,
            "status": "not_executed",
            "details": None,
            "error": None,
        }
        if selected is not None and "M10" not in selected:
            item["reason"] = "本次命令未选择 M10"
        elif cancel.is_set():
            item["status"] = "cancelled"
        else:
            print(f"M10 {identifier}", flush=True)
            try:
                item["details"] = action()
                item["status"] = "passed"
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = error_record(exc, identifier)
        checks.append(item)
        storage.writer.write("boundary_check", **item)
    return checks
