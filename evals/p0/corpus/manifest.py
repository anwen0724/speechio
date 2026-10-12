"""正式评分前核对标注、哈希、区间和类别覆盖。"""

import json
import re
from pathlib import Path

import numpy as np

from evals.p0.evaluation.scoring import normalize
from realtime_asr.audio.files import read_wav
from realtime_asr.errors import ASRError
from realtime_asr.runtime.models.manifest import sha256_file


def incomplete(message, operation="validate_corpus"):
    """缺少评分依据时明确终止，不忽略样本继续形成低错误率汇总。"""
    raise ASRError("RECORD_INCOMPLETE", operation, message)


def validate_intervals(intervals, count):
    """校验半开区间的范围、顺序与互不重叠。"""
    previous_end = 0
    for interval in intervals:
        if len(interval) != 2 or any(type(value) is not int for value in interval):
            incomplete("语音区间应为二元整数数组")
        start, end = interval
        if not 0 <= previous_end <= start < end <= count:
            incomplete("语音区间越界、重叠或无序")
        previous_end = end


def validate_coverage(samples, representatives, audio):
    """按 P0 类别检查数量及可机器复核的条件，听音仍为独立证据。"""
    by_category = {
        key: [item for item in samples if key in item["categories"]]
        for key in (
            "C01",
            "C02",
            "C03",
            "C04",
            "C05",
            "C06",
            "C07",
            "C08",
            "C09",
            "T01",
            "T02",
        )
    }
    minimum = {"C01": 3, "C07": 2, **{key: 1 for key in by_category if key not in ("C01", "C07")}}
    for category, count in minimum.items():
        if len(by_category[category]) < count:
            incomplete(f"{category} 的样本不足 {count} 个")
    if sum(not any(c.startswith("T") for c in item["categories"]) for item in samples) < 12:
        incomplete("基础样本不足 12 个")
    for category in ("C01", "C07"):
        if representatives.get(category) not in {
            item["sample_id"] for item in by_category[category]
        }:
            incomplete(f"{category} 代表样本未固定")
    speakers = {item.get("speaker_id") for item in by_category["C01"]}
    if None in speakers or len(speakers) < 2:
        incomplete("C01 应有至少两位可定位的说话人")
    for category, maximum in (("C01", 10), ("C07", 15)):
        for item in by_category[category]:
            if not 16000 <= item["sample_count"] <= maximum * 16000:
                incomplete(f"{category} 时长不符")
    for item in by_category["C02"]:
        speech_count = sum(end - start for start, end in item["speech_intervals"])
        if (
            item["sample_count"] < 480000
            or speech_count < 320000
            or len(normalize(item["reference_text"], "zh")) < 60
            or len(
                [part for part in re.split(r"[。！？.!?]+", item["reference_text"]) if part.strip()]
            )
            < 3
        ):
            incomplete("C02 的时长、有效语音或正文不足")
    for category in ("C08", "C09"):
        for item in by_category[category]:
            values = audio[item["sample_id"]]
            if item["sample_count"] != 48000 or item["reference_text"] or item["speech_intervals"]:
                incomplete(f"{category} 的时长或空标注不符")
            if bool(np.any(values)) != (category == "C09"):
                incomplete(f"{category} 的非零样本条件不符")
    for category, remainder in (("T01", 1), ("T02", 511)):
        for item in by_category[category]:
            if item["sample_count"] % 512 != remainder:
                incomplete(f"{category} 尾块长度不符")
    tail_has_speech = False
    for category in ("T01", "T02"):
        for item in by_category[category]:
            end = item["sample_count"]
            tail_start = end - end % 512
            for start, stop in item["speech_intervals"]:
                values = audio[item["sample_id"]][max(start, tail_start) : min(stop, end)]
                tail_has_speech |= bool(np.any(values))
    if not tail_has_speech:
        incomplete("至少一个尾块应包含标注语音的非零样本")


def load_corpus(path, *, allow_unreviewed=False):
    """校验清单并载入标准音频；未听音核实的清单只能用于探索。"""
    path = Path(path).resolve()
    try:
        corpus = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        incomplete(str(exc))
    if not corpus.get("corpus_version") or not corpus.get("samples"):
        incomplete("语料版本或样本缺失")
    ids, audio = set(), {}
    for item in corpus["samples"]:
        fields = {
            "sample_id",
            "audio_path",
            "audio_sha256",
            "sample_count",
            "sample_rate",
            "channels",
            "sample_width_bytes",
            "language",
            "categories",
            "reference_text",
            "speech_intervals",
            "source",
            "annotation_checked",
        }
        if not fields <= item.keys():
            incomplete(f"样本缺少字段：{sorted(fields - item.keys())}")
        identifier = item.get("sample_id")
        if not identifier or identifier in ids:
            incomplete("样本标识缺失或重复")
        ids.add(identifier)
        if item.get("annotation_checked") is not True and not allow_unreviewed:
            incomplete(f"{identifier} 尚未听音核实", "validate_annotation")
        if not allow_unreviewed and (item.get("speaker_id") or "").startswith("unverified-"):
            incomplete(f"{identifier} 的说话人尚未核实", "validate_annotation")
        file = (path.parent / item["audio_path"]).resolve()
        if not file.is_relative_to(path.parent):
            incomplete("语料路径超出清单目录")
        if not file.is_file() or sha256_file(file) != item["audio_sha256"]:
            incomplete(f"{identifier} 文件缺失或哈希改变", "validate_audio_hash")
        values = read_wav(file)
        if values.size != item["sample_count"]:
            incomplete(f"{identifier} 实际样本数不符")
        if (item["sample_rate"], item["channels"], item["sample_width_bytes"]) != (16000, 1, 2):
            incomplete(f"{identifier} 格式声明不符")
        if item["language"] not in ("zh", "en", "none"):
            incomplete(f"{identifier} 语言无效")
        if item["language"] != "none" and (
            not item["reference_text"] or not item["speech_intervals"]
        ):
            incomplete(f"{identifier} 缺少语音正文或区间")
        validate_intervals(item["speech_intervals"], values.size)
        if not all(
            name in item["source"] for name in ("source_id", "path_or_url", "sha256", "description")
        ):
            incomplete(f"{identifier} 来源记录不完整")
        source = item["source"]
        if source.get("local_path"):
            original = (path.parent / source["local_path"]).resolve()
            if (
                not original.is_relative_to(path.parent)
                or not original.is_file()
                or sha256_file(original) != source["sha256"]
            ):
                incomplete(f"{identifier} 原始素材缺失或改变", "validate_source_hash")
        audio[identifier] = values
    validate_coverage(corpus["samples"], corpus["representative_sample_id"], audio)
    validate_augmentations(corpus["samples"], corpus["representative_sample_id"], audio)
    return corpus, audio


def validate_augmentations(samples, representatives, audio):
    """复核固定增广的参数与可计算事实；参数正确不代替听音。"""
    by_id = {item["sample_id"]: item for item in samples}
    for item in samples:
        category = item["categories"][0]
        if category not in ("C03", "C04", "C05", "C06"):
            continue
        preparation = item.get("preparation")
        if not preparation or not preparation.get("method") or not preparation.get("parameters"):
            incomplete(f"{category} 缺少增广记录")
        parameters = preparation["parameters"]
        parent_id = preparation.get("parent_sample_id")
        if category != "C04" and parent_id != representatives["C01"]:
            incomplete(f"{category} 不是固定代表音频的增广")
        if parent_id not in by_id:
            incomplete(f"{category} 母样本不存在")
        parent = by_id[parent_id]
        if category == "C03":
            start, end = parameters.get("gain_range", [None, None])
            expected_start = parent["speech_intervals"][0][0]
            if (
                parameters.get("gain_factor") != 0.1
                or start != expected_start
                or not start < end <= min(start + 8000, parent["sample_count"])
            ):
                incomplete("C03 衰减区间或比例不符")
            expected = audio[parent_id].copy()
            expected[start:end] *= 0.1
            if expected.size != item["sample_count"] or not np.allclose(
                expected, audio[item["sample_id"]], atol=1 / 32768, rtol=0
            ):
                incomplete("C03 音频不符合衰减记录")
        elif category == "C04":
            intervals = item["speech_intervals"]
            if (
                parameters.get("pause_samples") != 4800
                or len(intervals) < 2
                or any(end - start < 16000 for start, end in intervals[:2])
            ):
                incomplete("C04 暂停或两侧有效语音不足")
            pause_start, pause_end = intervals[0][1], intervals[1][0]
            if pause_end - pause_start != 4800 or np.any(
                audio[item["sample_id"]][pause_start:pause_end]
            ):
                incomplete("C04 插入静音不符")
        elif category == "C05":
            if (
                parameters.get("speed_ratio") != 1.25
                or item["sample_count"] != (parent["sample_count"] * 4 + 4) // 5
            ):
                incomplete("C05 变速比例或实际样本数不符")
        elif parameters.get("snr_db") != 10 or not all(
            name in parameters
            for name in ("noise_sample_id", "noise_range", "noise_scale", "mix_scale", "seed")
        ):
            incomplete("C06 噪声与混合参数不完整")
