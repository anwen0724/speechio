import { Transcription } from '../features/transcription/Transcription';

/** 页面只划分区域；采集、协议和结果交互由转写功能负责。 */
export function App() {
  return (
    <main className="workspace">
      <header className="workspace-header">
        <span className="eyebrow">LOCAL SPEECH · SHARED SERVICE</span>
        <h1>Realtime ASR</h1>
        <p>打开麦克风，把语音变成文字。</p>
      </header>
      <Transcription />
    </main>
  );
}
