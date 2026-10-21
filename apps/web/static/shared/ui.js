/** 渲染已确认的两套布局；所有动态结果由服务端事件填充。 */
export function mount(debug) {
const $ = (selector) => document.querySelector(selector);
const icons = {
  mic: '<rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/>',
  file: '<path d="M14 2H5v20h14V7zM14 2v5h5M8 12h8M8 16h6"/>',
  settings: '<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="2"/><circle cx="15" cy="17" r="2"/>',
  copy: '<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M15 8V3H3v13h5"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  upload: '<path d="M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7M14 10v7"/>'
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[name]}</svg>`;

function selectField(id, title, options, help = '') {
  return `<label class="field"><span class="field-title">${title}</span>
    <select id="${id}">${options.map(option => `<option>${option}</option>`).join('')}</select>
    ${help ? `<small>${help}</small>` : ''}</label>`;
}

function numberField(id, title, value, help = '') {
  return `<label class="field"><span class="field-title">${title}</span>
    <input id="${id}" type="number" step="any" value="${value}" placeholder="不限制">
    ${help ? `<small>${help}</small>` : ''}</label>`;
}

function toggle(id, title, checked = false) {
  return `<label class="toggle"><span>${title}</span><input id="${id}" type="checkbox" ${checked ? 'checked' : ''}></label>`;
}

function runtimeFields(prefix = '') {
  return `<div class="field-pair" id="${prefix}runtime-fields">
    ${selectField(`${prefix}compute-type`, '计算精度', [], '配置默认沿用文件值；引擎默认交给运行库选择。')}
    ${numberField(`${prefix}device-index`, 'GPU 序号', '0', '从 0 开始；CPU 使用 0。')}
  </div>`;
}

function modelSettings() {
  return `${selectField('engine', '识别后端', [])}
    ${selectField('model', '模型', [], '模型来自服务端配置的预设。')}
    <div class="field-pair">
      ${selectField('language', '识别语言', ['中文', '自动检测', 'English'])}
      ${selectField('device', '运行设备', [])}
    </div>
    ${debug ? runtimeFields() : ''}
    ${debug ? `<div id="remote-runtime" hidden>${numberField('timeout', '远程请求超时（秒）', '30', '仅修改本次请求，不写回配置文件。')}</div>` : ''}
    <p class="help" id="compatibility">配置文件提供默认值，页面设置用于本次运行。</p>
    <p class="help" id="remote-note" hidden></p>`;
}

function sessionSettings() {
  return `<div data-live-only>
    ${selectField('segmentation', '分段方式', ['停顿后自动分段', '手动结束当前段'])}
    ${numberField('silence', '停顿多久后分段（秒）', '0.8', '例如 0.8：连续停顿约 0.8 秒后，将这句话归为一段。')}
    ${toggle('wake', '启用唤醒词')}
    <div id="wake-options" hidden>
      ${selectField('keyword', '已配置的唤醒词', [])}
      <p class="help">这里只展示预置选项，不代表可以输入任意词立即识别。</p>
    </div>
  </div>`;
}

function outputSettings() {
  return `${toggle('partials', '显示实时文字', true)}
    ${toggle('timestamps', '显示时间戳', true)}
    <div data-file-only hidden>
      ${numberField('max-seconds', '音频时长上限（秒）', '', '留空不限制；可填写任意正数。超过上限时拒绝处理，不截断音频。')}
    </div>`;
}

function debugSettings() {
  return `<aside class="config-panel"><h2>参数配置 <span class="badge">下一次运行生效</span></h2>
    <details open><summary>01　引擎与模型</summary>${modelSettings()}</details>
    <details><summary>02　实时识别</summary>
      ${selectField('realtime-mode', '实时处理方式', ['快照识别', '原生流式（取决于引擎）'], '快照：定时识别当前积累的音频；原生流式：引擎持续接收音频。')}
      ${selectField('realtime-model', '实时识别模型', ['共用主模型'])}
      ${toggle('retranscribe', '结束后用主模型重新识别')}
      <p class="help" id="completion-note">默认直接采用流式任务的最终结果。开启重识别会额外处理整段音频；云端可能增加费用。</p>
      <div id="realtime-runtime" hidden>
        ${selectField('realtime-device', '实时模型运行设备', [])}
        ${runtimeFields('realtime-')}
        <div id="realtime-remote-runtime" hidden>${numberField('realtime-timeout', '实时模型请求超时（秒）', '30', '沿用独立实时模型的默认值，可修改本次请求。')}</div>
      </div>
      <p class="help" id="realtime-runtime-note">共用主模型时，设备和精度也随主模型。</p>
      ${numberField('interval', '文字更新间隔（秒）', '0.2')}
      ${toggle('stabilize', '稳定临时文字', true)}
      <p class="help">稳定文字：减少已经显示的文字来回变化。只影响临时文字的显示，不修改最终结果。</p>
    </details>
    <details><summary>03　检测、分段与唤醒</summary>${sessionSettings()}
      ${numberField('vad', '语音检测灵敏度（0–1）', '0.4', '语音检测用于区分说话与静音。')}
      ${selectField('wake-backend', '唤醒后端', ['OpenWakeWord', 'Porcupine'])}
      ${numberField('wake-sensitivity', '唤醒灵敏度（0–1）', '0.5')}
      ${numberField('wake-timeout', '唤醒后等待说话（秒）', '5')}
    </details>
    <details><summary>04　候选与修正</summary>
      ${toggle('candidates', '启用候选结果')}
      ${toggle('correction', '保留音频以便修正')}
      <p class="help">候选是确认前的备选文本；修正是对已完成结果的后续更新。在右侧“结果对照”中查看。</p>
    </details>
    <details><summary>05　容量与结果</summary>${outputSettings()}
      ${numberField('buffer', '会话音频上限（采样点）', '480000', '16 kHz 单声道下，480000 个采样点约为 30 秒音频。')}
      ${numberField('pending', '待处理段数上限', '8')}
    </details>
    <details><summary>06　文件分块输入</summary>
      ${numberField('chunk', '每块音频时长（毫秒）', '100')}
      ${selectField('pace', '送入速度', ['按音频原速', '尽快送入'])}
      <p class="help">将同一文件分段送入，复现连续输入。每块音频都通过核心包的连续输入接口处理。</p>
    </details>
  </aside>`;
}

const tabs = `<nav class="tabs" aria-label="输入方式">
  <button data-mode="live" class="active" aria-pressed="true">${icon('mic')}实时转写</button>
  <button data-mode="file" aria-pressed="false">${icon('file')}文件转写</button>
  ${debug ? '<button data-mode="chunks" aria-pressed="false">文件分块</button>' : ''}
</nav>`;

function inputCard() {
  const bars = Array.from({ length: 67 }, (_, index) => {
    const height = 9 + Math.round(Math.abs(Math.sin(index * 1.8) * Math.cos(index * .23)) * 43);
    return `<i style="--height:${height}px;--delay:${-(index % 8) * .1}s"></i>`;
  }).join('');
  return `<section class="input-card">
    <div id="microphone-input">
      <div class="input-top"><label class="device">${icon('mic')}
        <select id="microphone" aria-label="浏览器麦克风"><option value="">默认麦克风</option></select>
      </label><span class="subtle">浏览器麦克风</span></div>
      <div class="wave" aria-hidden="true">${bars}</div>
    </div>
    <div id="file-input" hidden>
      <label class="upload" id="drop-zone">${icon('upload')}<span>拖入音频，或选择文件</span>
        <span class="subtle">上传到本机服务识别，处理结束后删除临时文件。</span>
        <span class="subtle">WAV、FLAC、MP3、AIFF、M4A、AAC、OGG、Opus、WebM、MP4；视频文件只读取第一条音轨</span>
        <input id="file-picker" type="file" accept=".wav,.flac,.mp3,.aiff,.aif,.m4a,.aac,.ogg,.opus,.webm,.mp4" aria-label="选择音频文件">
      </label>
      <div class="file-info">${icon('file')}<div><strong id="filename">尚未选择文件</strong><small id="file-meta">选择 WAV、FLAC 等音频文件</small></div></div>
      <audio id="player" controls hidden></audio>
    </div>
    <div class="transport"><span class="status" id="status" role="status">已就绪，可以开始</span>
      <button class="primary" id="start"><span class="record-dot"></span><span id="start-label">开始录音</span></button>
      <span class="timer" id="timer">00:00</span>
    </div>
    <div class="input-bottom"><span class="pill" id="input-policy">自动分段 · 停顿 0.8 秒</span>
      <button class="text-button" id="end-turn" disabled>结束当前段 →</button>

    </div>
    ${debug ? `<div class="input-bottom session-actions">
      <button class="quiet" id="cancel-turn" disabled>取消当前段</button>
      <button class="quiet" id="reset">重置会话</button>
      <button class="quiet" id="close-session">关闭会话</button>
      <button class="quiet" id="candidate" disabled>获取候选</button><button class="quiet" id="correct" disabled>修正上一段</button><span class="subtle" id="session-state">未连接</span>
    </div>` : ''}
    <div class="alert" id="error" role="alert" hidden></div>
  </section>`;
}

const resultActions = `<div class="actions">
  <button id="copy" title="复制文本">${icon('copy')}<span class="action-label">复制</span></button>
  <button id="export" title="导出 TXT">${icon('download')}<span class="action-label">导出</span></button>
  <button id="clear" title="清空显示，不结束会话">${icon('trash')}<span class="action-label">清空</span></button>
</div>`;

function resultCard() {
  return `<section class="${debug ? 'studio-results' : 'results'}">
    <div class="section-head">${debug ? `<nav class="result-tabs" aria-label="结果视图">
      <button id="text-tab" class="active" aria-pressed="true">转写结果</button>
      <button id="compare-tab" aria-pressed="false">结果对照</button></nav>` : '<h2>转写结果 <span id="recognition-location" class="badge">本地识别</span></h2>'}
      ${resultActions}
    </div>
    <div class="transcript" id="transcript"></div>
    ${debug ? '<div id="comparison" class="compare" hidden></div>' : ''}
    <div class="result-footer"><span id="result-count"></span><span id="model-summary">Whisper / base · 中文</span></div>
  </section>`;
}

const header = `<header class="topbar">
  <div class="brand"><span class="brand-mark">sı</span>${debug ? 'SpeechIO Studio' : 'SpeechIO'}<small>${debug ? 'DEVELOPER WORKSPACE' : '语音转写'}</small></div>
  <div class="prototype-tools"><span>本机运行</span><a id="other-page" href="${debug ? "/" : "/debug"}">${debug ? "语音转写" : "调试台"} ↗</a></div>
</header>`;

if (debug) {
  $('#app').innerHTML = header + `<main class="studio-main">
    <div class="studio-title"><div><p class="eyebrow">SPEECHIO / DEVELOPER WORKSPACE</p><h1>语音调试台</h1></div>
      <div class="session-actions"><button class="quiet" id="prepare">准备模型</button><button class="quiet" id="release">释放模型</button><span class="pill" id="model-state">模型未准备</span></div>
    </div>
    <div class="studio-grid">${debugSettings()}<div class="studio-content">
      <div class="toolbar">${tabs}<span class="subtle" id="audio-location">本机单任务 · 使用 speechio API</span></div>
      ${inputCard()}
      <div class="metrics">
        <div class="metric"><span>任务占用音频</span><strong id="metric-buffer">0 <small>秒</small></strong></div>
        <div class="metric"><span>待处理任务</span><strong id="metric-queue">0</strong></div>
        <div class="metric"><span>最终识别耗时</span><strong id="metric-latency">—</strong></div>
        <div class="metric"><span>已送入音频</span><strong id="metric-elapsed">0.0 <small>秒</small></strong></div>
      </div>
      ${resultCard()}
      <section class="console"><div class="section-head"><h2>事件记录 <span class="badge">实时事件</span></h2>
        <div class="actions"><select id="event-filter" aria-label="筛选事件"><option value="all">全部事件</option><option value="result">识别结果</option><option value="lifecycle">会话与模型</option><option value="error">错误</option></select><button id="export-events">导出 JSON</button></div>
      </div><div class="event-list" id="events"></div></section>
      <details class="developer"><summary>〈 / 〉配置与 Python 调用</summary><div class="developer-body">
        <div class="toolbar"><span class="help">下方配置来自当前控件；识别流程不在浏览器中执行。</span><button class="quiet" id="copy-config">复制配置</button></div>
        <p class="help">页面准备发送的参数</p><pre id="config-preview"></pre>
        <p class="help">服务端最近确认的模型配置（请求精度，不代表硬件实测精度）</p>
        <pre id="effective-config">尚未准备或开始</pre>
        <p class="help" id="code-note"></p><pre id="code-preview"></pre>
      </div></details>
    </div></div>
    <footer class="studio-footer"><span>参数 → 输入 → 识别结果 → 事件与清理状态</span><span>仅显示本连接的事件；模型由本机服务管理。</span></footer>
  </main>`;
} else {
  $('#app').innerHTML = header + `<main class="example-main">
    <div class="hero"><p class="eyebrow">SPACE FOR YOUR THOUGHTS</p><h1 id="page-title">说出来，记录下来。</h1><p id="page-subtitle">留心表达，文字交给 SpeechIO。</p></div>
    <div class="toolbar">${tabs}<button class="quiet" id="open-settings">${icon('settings')}识别设置</button></div>
    ${inputCard()}${resultCard()}
    <div class="toolbar" style="margin-top:24px"><span class="subtle">清空只移除文字，新录音会重新开始。</span><button class="text-button" id="new-session">＋ 新录音</button></div>
    <footer class="page-footer" id="audio-location">SpeechIO 示例应用 · 音频由本机服务处理</footer>
  </main>
  <div class="backdrop" id="backdrop" hidden></div>
  <section class="drawer" id="drawer" role="dialog" aria-modal="true" aria-label="识别设置" hidden>
    <div class="drawer-head"><h2>按你的方式识别</h2><button id="close-settings" aria-label="关闭设置">×</button></div>
    <p class="drawer-intro">调整下一次转写使用的配置。正在录音时，先结束录音再修改。</p>
    ${modelSettings()}<section class="settings-group"><h3>录音与分段</h3>${sessionSettings()}</section>
    <details class="settings-group" open><summary>更多设置</summary>${outputSettings()}</details>
    <div class="drawer-foot"><span class="subtle">配置保留在当前页面</span><button class="primary" id="done-settings">完成</button></div>
  </section>`;
}


}
