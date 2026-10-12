"""安装本次 wheel 到已有独立依赖环境，并从项目外验证实际包。"""

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path

from evals.p6.installed_check import invoke, run

PROBE = """import json
import numpy as np
import realtime_asr as sdk

class Engine(sdk.BaseTranscriptionEngine):
    def __init__(self):
        self.prepared = 0
        self.closed = 0

    @property
    def capabilities(self):
        return sdk.EngineCapabilities(backend='installed-custom')

    def prepare(self):
        self.prepared += 1

    def transcribe(self, audio, language=None, *, role='final', word_timestamps=False):
        assert len(audio) == 525 and audio.dtype == np.float32
        return sdk.RecognitionResult(text='installed custom', requested_language=language)

    def close(self):
        self.closed += 1

engine = Engine()
with sdk.Recognizer(engine=engine) as recognizer:
    result = recognizer.transcribe(np.zeros(525, np.int16), format=sdk.AudioFormat())
    assert result.text == 'installed custom'
assert engine.prepared == 1 and engine.closed == 1
engine = Engine()
with sdk.ASRRuntime(engine=engine) as runtime:
    session = runtime.create_session()
    session.start()
    session.feed_audio(np.zeros(525, np.int16))
    session.finish()
    assert session.get_event(timeout=5).result.text == 'installed custom'
    assert session.get_event(timeout=5).kind == 'completed'
    session.close()
assert engine.prepared == 1 and engine.closed == 1
print(json.dumps({
    'package': sdk.__file__, 'exports': len(sdk.__all__),
    'custom_sdk':True, 'custom_runtime':True, 'closed':engine.closed,
}))
"""

DUAL_PROBE = """import json
import numpy as np
from realtime_asr import (ASRRuntime, BaseTranscriptionEngine, EngineCapabilities,
    RealtimeConfig, RealtimeMergeConfig, RecognitionResult, SessionConfig, StreamingSession)

class Stream(StreamingSession):
    def __init__(self, text):
        self.text = text
        self.samples = 0
        self.closed = False

    def accept_audio(self, audio, sample_rate=None):
        self.samples += len(audio)

    def get_result(self):
        return RecognitionResult(text=self.text)

    def finish(self):
        return self.get_result()

    def close(self):
        self.closed = True

class Engine(BaseTranscriptionEngine):
    def __init__(self, text):
        self.text = text
        self.streams = []
        self.closed = 0

    @property
    def capabilities(self):
        return EngineCapabilities(backend='installed-dual', streaming=True)

    def create_streaming_session(self, language=None):
        stream = Stream(self.text)
        self.streams.append(stream)
        return stream

    def transcribe(self, audio, language=None, *, role='final', word_timestamps=False):
        assert len(audio) == 525
        return RecognitionResult(text='complete final')

    def close(self):
        self.closed += 1

accurate = Engine('speech system')
fast = Engine('speech system today')
with ASRRuntime(engine=accurate, realtime=RealtimeConfig(mode='native'),
        ultrafast=RealtimeConfig(mode='native', engine=fast)) as runtime:
    session = runtime.create_session(SessionConfig(realtime=True, merge=RealtimeMergeConfig()))
    session.start()
    session.feed_audio(np.arange(525, dtype=np.int16))
    session.finish()
    roles, merged = set(), []
    while True:
        event = session.get_event(timeout=5)
        assert event is not None and event.error is None
        if event.kind == 'realtime':
            roles.add(event.realtime_role)
            merged.append(event.merge.text)
        if event.kind == 'final':
            assert event.result.text == 'complete final'
        if event.kind == 'completed':
            break
    session.close()
assert roles == {'accurate', 'ultrafast'}
assert 'speech system today' in merged
assert accurate.closed == fast.closed == 1
assert all(stream.samples == 525 and stream.closed for engine in (accurate, fast)
    for stream in engine.streams)
print(json.dumps({'controlled_installed_dual': True, 'roles': sorted(roles)}))
"""

CANDIDATE_PROBE = (
    PROBE
    + """
engine = Engine()
with sdk.ASRRuntime(engine=engine, candidate=sdk.CandidateConfig()) as runtime:
    session = runtime.create_session(sdk.SessionConfig(
        completion_mode='candidate', retain_for_correction=True))
    unit = session.start()
    session.feed_audio(np.zeros(525, np.int16))
    session.resume(expected_audio_samples=525)
    session.request_candidate()
    candidate = session.get_event(timeout=5)
    assert candidate.candidate.request.resume_epoch == 1
    session.finish()
    final = session.get_event(timeout=5)
    assert final.optimization['candidate_reused']
    assert session.get_event(timeout=5).kind == 'completed'
    session.request_correction(unit)
    correction = session.get_event(timeout=5)
    assert correction.candidate.replaces_task_id == final.task_id
assert engine.prepared == engine.closed == 1
print(json.dumps({'controlled_installed_candidates': True, 'package': sdk.__file__}))
"""
)


WAKE_PROBE = (
    PROBE
    + """
class Vad:
    info = {'backend': 'installed-vad'}
    def detect(self, pcm, start_sample, *, recording=False):
        return sdk.VoiceActivity(True, True, True, True, .1)
    def reset(self): pass
class Wake:
    sample_rate = 16000
    frame_samples = 512
    def __init__(self): self.closed = 0
    def detect(self, pcm): return 'installed keyword'
    def reset(self): pass
    def close(self): self.closed += 1
detector = Wake()
with sdk.ASRRuntime(engine=Engine()) as runtime:
    session = runtime.create_auto_session(detector_factory=Vad,
        segmentation=sdk.SegmentationConfig(min_recording_duration=0),
        wake=sdk.WakeConfig(trim_duration=0), wake_detector_factory=lambda: detector)
    session.feed_audio(np.ones(525, np.int16))
    snapshot = session.audio_snapshot()
    assert snapshot.end_sample - snapshot.start_sample == 512
    session.finish_input()
    assert session.get_event(timeout=5).kind == 'final'
    assert session.get_event(timeout=5).kind == 'completed'
assert detector.closed == 1
print(json.dumps({'installed_wake_and_snapshot': True, 'package': sdk.__file__}))
"""
)


def main():
    parser = argparse.ArgumentParser(description="P8 wheel 项目外检查，复用已准备的三种依赖环境")
    parser.add_argument("--environments", type=Path, default=Path(".runtime/p6-installed"))
    parser.add_argument("--outside", type=Path, required=True)
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--wake", action="store_true")
    args = parser.parse_args()
    project = Path.cwd()
    outside = args.outside.resolve()
    if outside.is_relative_to(project):
        raise ValueError("独立检查目录应在项目外")
    environments = args.environments.resolve()
    wheel_path = project / "dist/realtime_asr-0.1.0-py3-none-any.whl"
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(project / ".runtime/uv-cache")
    env["UV_LINK_MODE"] = "copy"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    for mode in ("base", "client", "server"):
        interpreter = environments / mode / "Scripts/python.exe"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--reinstall",
                "--no-deps",
                str(wheel_path),
            ],
            env=env,
            check=True,
        )
    report = run(project, environments, outside)
    report["p8_api"] = json.loads(
        invoke(
            [
                str(environments / "base/Scripts/python.exe"),
                "-I",
                "-c",
                PROBE,
            ],
            outside,
            env,
        )
    )
    if args.dual:
        report["p8_1_api"] = json.loads(
            invoke(
                [str(environments / "base/Scripts/python.exe"), "-I", "-c", DUAL_PROBE],
                outside,
                env,
            )
        )
    if args.candidates:
        output = invoke(
            [str(environments / "base/Scripts/python.exe"), "-I", "-c", CANDIDATE_PROBE],
            outside,
            env,
        )
        report["p8_candidates_api"] = json.loads(output.splitlines()[-1])
    if args.wake:
        output = invoke(
            [str(environments / "base/Scripts/python.exe"), "-I", "-c", WAKE_PROBE], outside, env
        )
        report["p9_api"] = json.loads(output.splitlines()[-1])
    with zipfile.ZipFile(wheel_path) as wheel:
        names = [name for name in wheel.namelist() if name.startswith("realtime_asr/")]
        for name in names:
            if wheel.read(name) != (project / "src" / name).read_bytes():
                raise AssertionError(f"wheel 与实际源码／资源不一致：{name}")
        for required in ("backends/base.py", "backends/local.py", "runtime/tasks.py"):
            if f"realtime_asr/{required}" not in names:
                raise AssertionError(f"wheel 缺少 P8 模块：{required}")
    report["all_package_members_match_source"] = True
    report["dependency_profiles_reused"] = True
    report["package_reinstalled_from_current_wheel"] = True
    Path(
        "artifacts/p9/installed.json"
        if args.wake
        else "artifacts/p8-candidates/installed.json"
        if args.candidates
        else "artifacts/p8-1/installed.json"
        if args.dual
        else "artifacts/p8/installed.json"
    ).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("P8 installed wheel checks passed")


if __name__ == "__main__":
    main()
