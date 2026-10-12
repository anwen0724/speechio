import { useEffect, useRef, useState } from 'react';
import { asError, BrowserSession } from './connection';
import type { Mode, WireMessage } from './connection';
import './transcription.css';

type Status =
  'disconnected' | 'preparing' | 'ready' | 'recording' | 'draining' | 'closed' | 'error';
type FinalText = {
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

/** 一个会话协调三个区域；旧连接回调始终核对所属实例。 */
export function Transcription() {
  const [token, setToken] = useState('');
  const [mode, setMode] = useState<Mode>('auto');
  const [stableText, setStableText] = useState(false);
  const [status, setStatus] = useState<Status>('disconnected');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState<
    | {
        key: string;
        text: string;
        stable?: string;
        unstable?: string;
      }
    | undefined
  >(undefined);
  const [finals, setFinals] = useState<FinalText[]>([]);
  const session = useRef<BrowserSession | undefined>(undefined);
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
    if (message.type === 'closed') {
      setStatus('closed');
      owner.dispose().catch((error) => fail(owner, asError(error)));
      return;
    }
    const event = message.event;
    if (!event) return;
    const key = `${event.unit.session_id}:${event.unit.generation}:${event.unit.unit_id}`;
    if (event.kind === 'realtime' && event.result) {
      const stable = event.stabilization;
      setPreview({
        key,
        text: stable?.display_text ?? event.result.text,
        stable: stable?.stable_text,
        unstable: stable?.unstable_text,
      });
    }
    if (event.kind === 'final' && event.result) {
      const text = event.result.text;
      setFinals((current) => [
        ...current.filter((item) => item.key !== key),
        {
          key,
          text,
          generation: event.unit.generation,
          unit: event.unit.unit_id,
          completed: false,
          mismatch: event.finalization?.agrees_with_stable_prefix === false,
        },
      ]);
      setPreview((current) => (current?.key === key ? undefined : current));
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
    const previous = session.current;
    session.current = undefined;
    if (previous) await previous.dispose();
    const owner = new BrowserSession(
      mode,
      (message) => receive(owner, message),
      (error) => fail(owner, error),
    );
    session.current = owner;
    try {
      await owner.connect(token, stableText);
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

  async function copy() {
    try {
      await navigator.clipboard.writeText(finals.map((item) => item.text).join('\n'));
    } catch (error) {
      setError(`复制失败：${asError(error).message}`);
    }
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
            onChange={(event) => setMode(event.target.value === 'auto' ? 'auto' : 'manual')}
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
        <p className="hint">连接时申请麦克风权限。凭据仅保留在当前页面内存。</p>
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
      <section className="preview-panel" aria-label="实时观察">
        <div className="panel-heading">
          <h2>实时观察</h2>
          <span className="tag">{stableText ? '稳定与待定' : '可变预览'}</span>
        </div>
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
      </section>
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
    </>
  );
}
