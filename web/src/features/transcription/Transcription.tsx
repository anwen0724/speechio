import { useEffect, useRef, useState } from 'react';
import { asError, BrowserSession } from './connection';
import type { Mode, RealtimeMerge, WireMessage } from './connection';
import type { CandidateResult, CompletionMode, UnitRef } from './connection';
import { CandidatePanel } from './CandidatePanel';
import type { CandidateAction } from './CandidatePanel';
import { RuntimeInfo, Timeline } from './RuntimePanel';
import type { TimelineEntry } from './RuntimePanel';
import type { WakeStatus } from './connection';
import './transcription.css';

type Status =
  'disconnected' | 'preparing' | 'ready' | 'recording' | 'draining' | 'closed' | 'error';
type FinalText = {
  taskId?: string | null;
  key: string;
  text: string;
  generation: number;
  unit: number;
  completed: boolean;
  mismatch: boolean;
};
const labels: Record<Status, string> = {
  disconnected: '未连接',
  preparing: '准备中',
  ready: '已连接',
  recording: '采集中',
  draining: '正在排空',
  closed: '已关闭',
  error: '错误',
};
const mergeLabels: Record<RealtimeMerge['status'], string> = {
  waiting_for_accurate: '等待准确正文，极速原文单独展示。',
  accurate_only: '显示准确正文，尚无可追加尾部。',
  exact: '已通过精确锚点补充尾部。',
  bootstrap_exact: '初始英文词精确匹配，已补充尾部。',
  fuzzy: '受限模糊匹配已确认。',
  soft_frontier: '单调前沿已确认尾部。',
  held_fuzzy_pending: '等待模糊匹配确认，保留当前正文。',
  held_no_anchor: '极速结果没有锚点，保留当前正文。',
  held_fast_conflict: '极速尾部发生冲突，保留已经显示的部分。',
  accurate_advanced_no_anchor: '准确正文已更新，旧尾部已清除。',
  stale_ignored: '旧观察已忽略。',
};

/** 一个会话协调三个区域；旧连接回调始终核对所属实例。 */
export function Transcription() {
  const [token, setToken] = useState('');
  const [mode, setMode] = useState<Mode>('auto');
  const [stableText, setStableText] = useState(false);
  const [dualRealtime, setDualRealtime] = useState(false);
  const [completionMode, setCompletionMode] = useState<CompletionMode>('final');
  const [retainForCorrection, setRetainForCorrection] = useState(false);
  const [candidateAvailable, setCandidateAvailable] = useState(false);
  const [wake, setWake] = useState(false);
  const [wakeStatus, setWakeStatus] = useState<WakeStatus>();
  const [levelOwner, setLevelOwner] = useState<BrowserSession>();
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [candidate, setCandidate] = useState<CandidateResult>();
  const [correctionTarget, setCorrectionTarget] = useState<UnitRef>();
  const [status, setStatus] = useState<Status>('disconnected');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState<
    | {
        key: string;
        text: string;
        stable?: string;
        unstable?: string;
        merge?: RealtimeMerge;
      }
    | undefined
  >(undefined);
  const [finals, setFinals] = useState<FinalText[]>([]);
  const session = useRef<BrowserSession | undefined>(undefined);
  const latestUnit = useRef<UnitRef | undefined>(undefined);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      const owner = session.current;
      session.current = undefined;
      owner?.dispose().catch((error) => console.error('释放采集失败', error));
    };
  }, []);

  function fail(owner: BrowserSession, reason: Error) {
    if (!mounted.current || session.current !== owner) return;
    setError(reason.message);
    setStatus('error');
    setBusy(false);
  }

  function receive(owner: BrowserSession, message: WireMessage) {
    if (!mounted.current || session.current !== owner) return;
    const eventSummary = message.event
      ? `${message.event.kind} · 代次 ${message.event.unit.generation} · 单元 ${message.event.unit.unit_id}`
      : (message.command ?? message.type);
    setTimeline((current) => [
      ...current.slice(-99),
      { id: crypto.randomUUID(), time: new Date().toLocaleTimeString(), text: eventSummary },
    ]);
    if (message.wake_status) setWakeStatus(message.wake_status);
    if (message.command === 'start' && message.unit) latestUnit.current = message.unit;
    if (message.type === 'ready') setCandidateAvailable(message.candidate?.enabled ?? false);
    if (message.resume) {
      setPreview(undefined);
      setCandidate(undefined);
    }
    if (message.type === 'closed') {
      setStatus('closed');
      setCorrectionTarget(undefined);
      owner.dispose().catch((error) => fail(owner, asError(error)));
      return;
    }
    const event = message.event;
    if (!event) return;
    const key = `${event.unit.session_id}:${event.unit.generation}:${event.unit.unit_id}`;
    if (
      event.kind === 'realtime' &&
      (!latestUnit.current ||
        event.unit.generation > latestUnit.current.generation ||
        (event.unit.generation === latestUnit.current.generation &&
          event.unit.unit_id > latestUnit.current.unit_id))
    )
      latestUnit.current = event.unit;
    const latest = latestUnit.current;
    const isLatestUnit =
      !latest ||
      (latest.session_id === event.unit.session_id &&
        latest.generation === event.unit.generation &&
        latest.unit_id === event.unit.unit_id);
    if (event.candidate && isLatestUnit) setCandidate(event.candidate);
    if (event.kind === 'correction' && event.candidate?.text) {
      const correction = event.candidate;
      setFinals((current) =>
        current.map((item) =>
          item.key === key && item.taskId === correction.replaces_task_id
            ? { ...item, text: correction.text, mismatch: false }
            : item,
        ),
      );
    }
    if (event.kind === 'realtime' && event.result) {
      const stable = event.stabilization;
      const rawText = event.result.text;
      setPreview((current) => ({
        key,
        text: event.merge?.text ?? stable?.display_text ?? rawText,
        stable:
          stable?.stable_text ??
          (event.realtime_role === 'ultrafast' && current?.key === key
            ? current.stable
            : undefined),
        unstable:
          stable?.unstable_text ??
          (event.realtime_role === 'ultrafast' && current?.key === key
            ? current.unstable
            : undefined),
        merge: event.merge ?? undefined,
      }));
    }
    if (event.kind === 'final' && event.result) {
      const text = event.result.text;
      setFinals((current) => [
        ...current.filter((item) => item.key !== key),
        {
          taskId: event.task_id,
          key,
          text,
          generation: event.unit.generation,
          unit: event.unit.unit_id,
          completed: false,
          mismatch: event.finalization?.agrees_with_stable_prefix === false,
        },
      ]);
      setPreview((current) => (current?.key === key ? undefined : current));
      if (retainForCorrection && event.task_id && isLatestUnit) setCorrectionTarget(event.unit);
    }
    if (event.kind === 'completed') {
      setFinals((current) =>
        current.map((item) => (item.key === key ? { ...item, completed: true } : item)),
      );
    }
    // 任务或实时观察失败不等于连接失败，保留其他单元与最终路径。
    if (event.error) setError(`${event.error.code}: ${event.error.message}`);
  }

  async function connect() {
    setBusy(true);
    setError('');
    setStatus('preparing');
    setCorrectionTarget(undefined);
    setCandidate(undefined);
    setCandidateAvailable(false);
    setWakeStatus(undefined);
    latestUnit.current = undefined;
    const previous = session.current;
    session.current = undefined;
    if (previous) await previous.dispose();
    const owner = new BrowserSession(
      mode,
      (message) => receive(owner, message),
      (error) => fail(owner, error),
    );
    session.current = owner;
    setLevelOwner(owner);
    try {
      await owner.connect(token, {
        stableText,
        dualRealtime,
        completionMode,
        retainForCorrection,
        wake,
      });
      if (mounted.current && session.current === owner) setStatus('ready');
    } catch (error) {
      fail(owner, asError(error));
    } finally {
      if (mounted.current && session.current === owner) setBusy(false);
    }
  }

  async function act(action: 'start' | 'finish' | 'cancel' | 'reset' | 'close') {
    const owner = session.current;
    if (!owner) return;
    setBusy(true);
    setError('');
    if (action === 'close') setStatus('draining');
    if (action === 'start' || action === 'reset' || action === 'cancel') {
      setCorrectionTarget(undefined);
      setCandidate(undefined);
    }
    try {
      await owner[action]();
      if (mounted.current && session.current === owner) {
        if (action !== 'close') setStatus(action === 'start' ? 'recording' : 'ready');
        if (action === 'cancel' || action === 'reset') setPreview(undefined);
      }
    } catch (error) {
      fail(owner, asError(error));
      await owner.dispose();
    } finally {
      if (mounted.current && session.current === owner) setBusy(false);
    }
  }

  async function candidateAction(action: CandidateAction) {
    const owner = session.current;
    if (!owner) return;
    setBusy(true);
    setError('');
    try {
      if (action.kind === 'candidate') await owner.requestCandidate(action.source, action.hint);
      else if (action.kind === 'resume') await owner.resume();
      else if (correctionTarget) await owner.requestCorrection(correctionTarget);
    } catch (error) {
      // 控制拒绝不销毁仍在采集的连接，保留输入供继续或正常结束。
      if (session.current === owner && mounted.current) setError(asError(error).message);
    } finally {
      if (session.current === owner && mounted.current) setBusy(false);
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(finals.map((item) => item.text).join('\n'));
    } catch (error) {
      setError(`复制失败：${asError(error).message}`);
    }
  }

  function clearDisplay() {
    setPreview(undefined);
    setCandidate(undefined);
    setFinals([]);
    setTimeline([]);
    setCorrectionTarget(undefined);
  }

  const connected = status === 'ready' || status === 'recording';
  return (
    <>
      <section className="connection-panel" aria-label="连接与采集">
        <div className="panel-heading">
          <h2>连接与采集</h2>
          <span className={`status ${status}`}>{labels[status]}</span>
        </div>
        <label className="field">
          接入模式
          <select
            value={mode}
            disabled={connected || busy || status === 'draining'}
            onChange={(event) => {
              const next = event.target.value === 'auto' ? 'auto' : 'manual';
              setMode(next);
              if (next === 'manual') setWake(false);
            }}
          >
            <option value="auto">自动分句</option>
            <option value="manual">手动结束</option>
          </select>
        </label>
        <label className="field">
          Token
          <input
            type="password"
            autoComplete="off"
            value={token}
            disabled={connected || busy}
            onChange={(event) => setToken(event.target.value)}
            placeholder="回环服务可留空"
          />
        </label>
        <label className="stable-option">
          <input
            type="checkbox"
            checked={stableText}
            disabled={connected || busy}
            onChange={(event) => setStableText(event.target.checked)}
          />
          稳定文本
        </label>
        <label className="stable-option">
          <input
            type="checkbox"
            checked={dualRealtime}
            disabled={connected || busy}
            onChange={(event) => setDualRealtime(event.target.checked)}
          />
          双实时合并
        </label>
        <p className="hint">连接时申请麦克风权限。凭据仅保留在当前页面内存。</p>
        <label className="stable-option">
          <input
            type="checkbox"
            checked={wake}
            disabled={connected || busy || mode !== 'auto'}
            onChange={(event) => setWake(event.target.checked)}
          />
          等待唤醒词
        </label>
        <RuntimeInfo
          owner={levelOwner}
          ready={connected || status === 'draining'}
          wake={wakeStatus}
        />
        <label className="field">
          完成策略
          <select
            aria-label="完成策略"
            value={completionMode}
            disabled={connected || busy}
            onChange={(event) =>
              setCompletionMode(event.target.value === 'candidate' ? 'candidate' : 'final')
            }
          >
            <option value="final">最终模型</option>
            <option value="candidate">候选模型</option>
          </select>
        </label>
        <label className="stable-option">
          <input
            type="checkbox"
            checked={retainForCorrection}
            disabled={connected || busy}
            onChange={(event) => setRetainForCorrection(event.target.checked)}
          />
          保留最近整轮供修正
        </label>
        <div className="actions">
          <button
            className="primary"
            disabled={connected || busy || status === 'draining'}
            onClick={() => {
              connect().catch((error) => setError(asError(error).message));
            }}
          >
            连接
          </button>
          <button
            disabled={status !== 'ready' || busy}
            onClick={() => {
              act('start').catch((error) => setError(asError(error).message));
            }}
          >
            开始采集
          </button>
          <button
            disabled={status !== 'recording' || busy}
            onClick={() => {
              act('finish').catch((error) => setError(asError(error).message));
            }}
          >
            结束当前输入
          </button>
          <button
            disabled={!connected || busy}
            onClick={() => {
              act('cancel').catch((error) => setError(asError(error).message));
            }}
          >
            取消当前单元
          </button>
          <button
            disabled={!connected || busy}
            onClick={() => {
              act('reset').catch((error) => setError(asError(error).message));
            }}
          >
            重置会话
          </button>
          <button
            disabled={!connected || busy}
            onClick={() => {
              act('close').catch((error) => setError(asError(error).message));
            }}
          >
            断开
          </button>
        </div>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        <p className="hint">取消舍弃当前输入；重置进入新代次。断开会先排空本连接的结果。</p>
      </section>
      <section
        className={`preview-panel${dualRealtime ? ' dual-preview-panel' : ''}`}
        aria-label="实时观察"
      >
        <div className="panel-heading">
          <h2>实时观察</h2>
          <span className="tag">
            {dualRealtime ? '双路与合并' : stableText ? '稳定与待定' : '可变预览'}
          </span>
        </div>
        {dualRealtime ? (
          <div className="dual-observations">
            <h3>准确正文</h3>
            <p className="preview" aria-label="准确实时正文">
              {preview?.merge?.accurate_text || '等待准确路结果。'}
            </p>
            {stableText && preview?.stable !== undefined && (
              <>
                <h3>准确路稳定与待定</h3>
                <p>
                  <strong className="stable-text" aria-label="稳定前缀">
                    {preview.stable}
                  </strong>
                  <span className="unstable-text" aria-label="待定尾部">
                    {preview.unstable}
                  </span>
                </p>
              </>
            )}
            <h3>极速原文</h3>
            <p aria-label="极速实时原文">{preview?.merge?.ultrafast_text || '等待极速路结果。'}</p>
            <h3>合并文本</h3>
            <p className="preview" aria-label="合并实时文本">
              {preview?.merge?.text || '等待准确正文与可信锚点。'}
            </p>
            <p className="hint" aria-label="合并状态">
              {preview?.merge ? mergeLabels[preview.merge.status] : '开始采集后显示两路结果。'}
            </p>
          </div>
        ) : (
          <p className={preview ? 'preview' : 'empty'}>
            {preview?.stable !== undefined ? (
              <>
                <strong className="stable-text" aria-label="稳定前缀">
                  {preview.stable}
                </strong>
                <span className="unstable-text" aria-label="待定尾部">
                  {preview.unstable}
                </span>
              </>
            ) : (
              preview?.text || '开始说话后，这里显示实时识别。'
            )}
          </p>
        )}
      </section>
      <CandidatePanel
        value={candidate}
        available={candidateAvailable}
        busy={busy}
        recording={status === 'recording'}
        canResume={mode === 'manual' && status === 'recording'}
        canCorrect={connected && correctionTarget !== undefined}
        onAction={(action) => {
          candidateAction(action).catch((error) => setError(asError(error).message));
        }}
      />
      <section className="final-panel" aria-label="最终结果">
        <div className="panel-heading">
          <h2>
            最终结果 <span className="count">{finals.length}</span>
          </h2>
          <button
            className="text-button"
            disabled={!finals.length}
            onClick={() => {
              copy().catch((error) => setError(asError(error).message));
            }}
          >
            复制文本
          </button>
          <button className="text-button" onClick={clearDisplay}>
            清除展示
          </button>
        </div>
        {!finals.length && <p className="empty">结束输入后，最终文本会保留在这里。</p>}
        <ol className="finals">
          {finals.map((item) => (
            <li key={item.key}>
              <span className="result-meta">
                代次 {item.generation} · 单元 {item.unit} · {item.completed ? '已完成' : '等待完成'}
              </span>
              <p>{item.text || '（无识别正文）'}</p>
              {item.mismatch && (
                <p className="hint">最终正文与实时稳定前缀不同，以上保留最终识别结果。</p>
              )}
            </li>
          ))}
        </ol>
      </section>
      <Timeline entries={timeline} />
    </>
  );
}
