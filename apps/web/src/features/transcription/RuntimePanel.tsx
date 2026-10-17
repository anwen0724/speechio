import { useSyncExternalStore } from 'react';
import type { BrowserSession, WakeStatus } from './connection';

const idleSubscribe = () => () => {};
const idleLevel = () => 0;
const wakeLabels: Record<WakeStatus['state'], string> = {
  disabled: '直接识别',
  grace: '可直接说话',
  awaiting_wake: '等待唤醒词',
  awaiting_speech: '已唤醒，等待正文',
  recording: '正在识别正文',
  followup: '继续说话窗口',
  finished: '输入已结束',
  failed: '门控失败',
  closed: '门控已关闭',
};
export type TimelineEntry = { id: string; time: string; text: string };

/** 音量只订阅当前采集流，不推动其他识别区域重复渲染。 */
export function RuntimeInfo({
  owner,
  ready,
  wake,
}: {
  owner: BrowserSession | undefined;
  ready: boolean;
  wake: WakeStatus | undefined;
}) {
  const level = useSyncExternalStore(
    owner?.subscribeLevel ?? idleSubscribe,
    owner?.getAudioLevel ?? idleLevel,
  );
  return (
    <div className="runtime-info" aria-label="运行状态">
      <p className="hint">{ready ? '宿主模型已就绪' : '等待服务就绪'}</p>
      <label>
        采集音量 <meter aria-label="采集音量" min={0} max={1} value={level} />
      </label>
      <p aria-label="门控状态">
        {wake ? wakeLabels[wake.state] : '未取得门控状态'}
        {wake?.keyword ? ` · ${wake.keyword}` : ''}
      </p>
    </div>
  );
}

export function Timeline({ entries }: { entries: TimelineEntry[] }) {
  return (
    <section className="timeline-panel" aria-label="会话时间线">
      <div className="panel-heading">
        <h2>会话时间线</h2>
        <span className="tag">最近 100 条</span>
      </div>
      <ol className="timeline">
        {entries.map((entry) => (
          <li key={entry.id}>
            <time>{entry.time}</time> · {entry.text}
          </li>
        ))}
      </ol>
      {!entries.length && <p className="empty">连接和识别事件会显示在这里。</p>}
    </section>
  );
}
