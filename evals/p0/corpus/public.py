"""准备 FLEURS 公开录音及增广草稿；不将草稿标为人工已核实。"""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from evals.p0.corpus.augment import (
    add_noise,
    draft_intervals,
    pause_between,
    quiet_opening,
    speed_up,
    tail_variant,
    write_pcm16,
)
from realtime_asr.runtime.models.manifest import sha256_file


def download_rows(folder, config, indexes):
    """保留公开来源索引，下载所选原录音；不使用识别结果生成正文。"""
    import requests
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    metadata_file = folder / f"source-{config}.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    else:
        response = requests.get(
            "https://datasets-server.huggingface.co/first-rows",
            params={"dataset": "google/fleurs", "config": config, "split": "validation"},
            timeout=(15, 30),
        )
        response.raise_for_status()
        metadata = response.json()
        metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    recordings = []
    for index in indexes:
        row = metadata["rows"][index]["row"]
        url = row["audio"][0]["src"]
        raw = folder / f"source-{config}-{row['id']}.wav"
        if not raw.exists():
            response = requests.get(url, timeout=(15, 30))
            response.raise_for_status()
            raw.write_bytes(response.content)
        rate, values = wavfile.read(raw)
        original_format = {
            "sample_rate": int(rate),
            "channels": 1,
            "dtype": str(values.dtype),
            "sample_count": int(values.size),
        }
        if values.ndim != 1:
            raise ValueError("公开素材不是单声道")
        if np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.float32) / (2 ** (values.dtype.itemsize * 8 - 1))
        else:
            values = values.astype(np.float32)
        if rate != 16000:
            values = resample_poly(values, 16000, rate).astype(np.float32)
        recordings.append(
            (
                row,
                values,
                {
                    "source_id": f"google/fleurs:{config}:validation:{row['id']}",
                    "path_or_url": url.split("?")[0],
                    "sha256": sha256_file(raw),
                    "local_path": raw.name,
                    "source_format": original_format,
                    "conversion": {
                        "method": "scale_integer_or_preserve_float_then_resample_poly_then_pcm16",
                        "source_sample_rate": int(rate),
                        "target_sample_rate": 16000,
                        "peak_normalization": False,
                    },
                    "description": "FLEURS 公开录音及原始正文；本项目听音和边界核实待完成。",
                },
            )
        )
    return recordings


def prepare_public_corpus(folder):
    """显式准备 12 个基础样本与两个尾块变体，已有清单保留不覆盖。"""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.json"
    if manifest.exists():
        return manifest
    zh = download_rows(folder, "cmn_hans_cn", [1, 2, 3, 0, 4, 5])
    en = download_rows(folder, "en_us", [0, 2])
    samples = []

    def save(
        identifier,
        category,
        values,
        text,
        intervals,
        source,
        *,
        language="zh",
        preparation=None,
        speaker_id=None,
    ):
        filename = f"{identifier}.wav"
        write_pcm16(folder / filename, values)
        samples.append(
            {
                "sample_id": identifier,
                "audio_path": filename,
                "audio_sha256": sha256_file(folder / filename),
                "language": language,
                "categories": [category],
                "reference_text": text,
                "sample_rate": 16000,
                "channels": 1,
                "sample_width_bytes": 2,
                "sample_count": values.size,
                "speech_intervals": intervals,
                "source": source,
                "annotation_checked": False,
                "preparation": preparation or source.get("conversion"),
                "speaker_id": speaker_id,
            }
        )

    for index, (row, values, source) in enumerate(zh[:3], 1):
        # 源数据只有性别标签，尚不冒充精确说话人标识；人工核实后应补充。
        save(
            f"C01-{index}",
            "C01",
            values,
            row["raw_transcription"],
            draft_intervals(values),
            source,
            speaker_id=f"unverified-gender-group-{row['gender']}",
        )
    long_audio = np.concatenate([values for _, values, _ in zh[3:]])
    offset, long_ranges = 0, []
    for _, values, _ in zh[3:]:
        long_ranges.extend(
            [[start + offset, end + offset] for start, end in draft_intervals(values)]
        )
        offset += values.size
    long_source = {**zh[3][2], "description": "三份公开录音依次拼接；来源详见 preparation。"}
    save(
        "C02-1",
        "C02",
        long_audio,
        "".join(row["raw_transcription"] for row, _, _ in zh[3:]),
        long_ranges,
        long_source,
        preparation={
            "parent_sample_id": None,
            "method": "concatenate",
            "parameters": {"sources": [s for _, _, s in zh[3:]]},
        },
    )
    row, parent, source = zh[0]
    intervals = draft_intervals(parent)
    noise_seed = 12012
    noise = (
        np.random.default_rng(noise_seed)
        .normal(0, 0.05, max(parent.size, 48000))
        .astype(np.float32)
    )
    write_pcm16(folder / "noise-source.wav", noise)
    noise_source = {
        "source_id": "gaussian-noise-12012",
        "path_or_url": "noise-source.wav",
        "sha256": sha256_file(folder / "noise-source.wav"),
        "description": "固定种子的非语音噪声。",
    }
    transforms = [
        (
            "C03-1",
            "C03",
            quiet_opening(parent, intervals),
            row["raw_transcription"],
            "quiet_opening",
        ),
        (
            "C04-1",
            "C04",
            pause_between(parent, intervals),
            row["raw_transcription"] * 2,
            "insert_pause",
        ),
        ("C05-1", "C05", speed_up(parent, intervals), row["raw_transcription"], "resample_poly"),
        (
            "C06-1",
            "C06",
            add_noise(
                parent, intervals, noise, noise_sample_id="gaussian-noise-12012", seed=noise_seed
            ),
            row["raw_transcription"],
            "mix_noise",
        ),
        (
            "T01",
            "T01",
            tail_variant(parent, intervals, 1),
            row["raw_transcription"],
            "adjust_silence",
        ),
        (
            "T02",
            "T02",
            tail_variant(parent, intervals, 511),
            row["raw_transcription"],
            "adjust_silence",
        ),
    ]
    for identifier, category, (values, ranges, parameters), text, method in transforms:
        save(
            identifier,
            category,
            values,
            text,
            ranges,
            source,
            preparation={
                "parent_sample_id": "C01-1",
                "method": method,
                "parameters": parameters,
            },
        )
    for index, (row, values, source) in enumerate(en, 1):
        save(
            f"C07-{index}",
            "C07",
            values,
            row["raw_transcription"],
            draft_intervals(values),
            source,
            language="en",
        )
    save(
        "C08-1",
        "C08",
        np.zeros(48000, np.float32),
        "",
        [],
        {
            "source_id": "zero-pcm",
            "path_or_url": "C08-1.wav",
            "sha256": None,
            "description": "生成 48000 个零样本，没有外部原文件。",
        },
        language="none",
    )
    save("C09-1", "C09", noise[:48000], "", [], noise_source, language="none")
    content = {
        "corpus_version": f"p0-public-draft-{uuid4().hex[:8]}",
        "representative_sample_id": {"C01": "C01-1", "C07": "C07-1"},
        "samples": samples,
        "review_status": "pending_human_listening",
        "source": "https://huggingface.co/datasets/google/fleurs",
    }
    manifest.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
