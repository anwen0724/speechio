import { useState } from 'react';
import type { CandidateResult } from './connection';

export type CandidateAction =
  | { kind: 'candidate'; source: 'tail' | 'whole'; hint: boolean }
  | { kind: 'resume' }
  | { kind: 'correct' };

/** 候选是独立观察；面板不决定最终任务或模型归属。 */
export function CandidatePanel({
  value,
  available,
  busy,
  recording,
  canResume,
  canCorrect,
  onAction,
}: {
  value: CandidateResult | undefined;
  available: boolean;
  busy: boolean;
  recording: boolean;
  canResume: boolean;
  canCorrect: boolean;
  onAction: (action: CandidateAction) => void;
}) {
  const [hint, setHint] = useState(false);
  return (
    <section className="candidate-panel" aria-label="候选与继续">
      <div className="panel-heading">
        <h2>候选与继续</h2>
        <span className="tag">{available ? '已启用' : '宿主未启用'}</span>
      </div>
      <div className="actions">
        {(['whole', 'tail'] as const).map((source) => (
          <button
            key={source}
            disabled={!available || !recording || busy}
            onClick={() => onAction({ kind: 'candidate', source, hint })}
          >
            {source === 'whole' ? '整轮候选' : '尾部修复'}
          </button>
        ))}
        <button
          disabled={!available || !canResume || busy}
          onClick={() => onAction({ kind: 'resume' })}
        >
          同轮继续
        </button>
        <button
          disabled={!available || !canCorrect || busy}
          onClick={() => onAction({ kind: 'correct' })}
        >
          修正最近整轮
        </button>
      </div>
      <label className="stable-option">
        <input type="checkbox" checked={hint} onChange={(event) => setHint(event.target.checked)} />
        提示性候选
      </label>
      <div className="candidate-content" aria-label="候选正文">
        {value ? (
          <>
            <p>{value.text || '（无候选正文）'}</p>
            <p className="hint">
              {value.request.kind} · {value.status} · 样本 {value.request.start_sample}—
              {value.request.end_sample} · 继续代次 {value.request.resume_epoch}
            </p>
            <p className="hint">
              {value.current ? '发布时覆盖最新输入' : '发布时输入已继续'} · 模型调用{' '}
              {value.attempts} 次
            </p>
            {value.model_text !== value.text && (
              <p className="hint">模型原文：{value.model_text || '（空）'}</p>
            )}
          </>
        ) : (
          <p className="empty">候选在这里单独展示。修正成功后更新对应的最终正文。</p>
        )}
      </div>
    </section>
  );
}
