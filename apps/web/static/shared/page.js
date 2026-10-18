/** 共享页面行为。B 保持简洁；C 额外订阅事件、容量与模型控制。 */
import { mount } from './ui.js';
import { SpeechClient, request } from './client.js';
import { Microphone } from './audio.js';

export async function startPage(debug) {
  mount(debug);
  const $ = selector => document.querySelector(selector);
  const state = {
    mode: 'live', phase: 'idle', client: null, microphone: null, file: null,
    fileUrl: null, token: null, rows: new Map(), events: [], comparison: {},
    receivedSeconds: 0, config: null, publicConfig: null, completionMessage: null,
  };
  const busy = () => ['preparing', 'recording', 'draining', 'stopping', 'closing'].includes(state.phase);
  const number = (id, fallback) => $(id)?.value === '' ? fallback : Number($(id)?.value ?? fallback);
  const checked = (id, fallback = false) => $(id)?.checked ?? fallback;
  const clock = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

  function options(element, entries, preferred) {
    element.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
    if (entries.some(([value]) => value === preferred)) element.value = preferred;
  }

  function showError(error) {
    const panel = $('#error');
    panel.hidden = false;
    panel.textContent = error.message || String(error);
  }

  function toast(message) {
    $('.toast')?.remove();
    const element = document.createElement('div');
    element.className = 'toast';
    element.setAttribute('role', 'status');
    element.textContent = message;
    document.body.append(element);
    setTimeout(() => element.remove(), 2500);
  }

  function log(message) {
    if (!debug || message.type === 'metrics') return;
    state.events.push({ received_at: new Date().toISOString(), ...message });
    if (state.events.length > 500) state.events.shift();
    renderEvents();
  }

  function renderEvents() {
    const filter = $('#event-filter').value;
    $('#events').replaceChildren();
    for (const event of state.events) {
      const category = event.type === 'error' || event.error ? 'error' : event.result ? 'result' : 'lifecycle';
      if (filter !== 'all' && filter !== category) continue;
      const row = document.createElement('details');
      row.className = `event-row ${category === 'error' ? 'error' : ''}`;
      const summary = document.createElement('summary');
      for (const [tag, text] of [
        ['span', event.received_at.slice(11, 19)], ['b', event.kind || event.type],
        ['span', event.result?.text || event.message || event.command || ''],
      ]) {
        const element = document.createElement(tag);
        element.textContent = text;
        summary.append(element);
      }
      const detail = document.createElement('pre');
      detail.textContent = JSON.stringify(event, null, 2);
      row.append(summary, detail);
      $('#events').append(row);
    }
  }

  function renderResults() {
    $('#transcript').replaceChildren();
    const rows = [...state.rows.values()];
    if (!rows.length) $('#transcript').innerHTML = '<div class="empty">这里会留下你的文字。<br>开始录音，或选择一个音频文件。</div>';
    for (const row of rows) {
      if (row.partial && !checked('#partials', true)) continue;
      const element = document.createElement('div');
      element.className = `utterance${row.partial ? ' partial' : ''}`;
      const stamp = document.createElement('span');
      stamp.className = 'stamp';
      stamp.textContent = row.seconds == null ? '—' : clock(row.seconds);
      const text = document.createElement('p');
      text.textContent = row.text;
      if (checked('#timestamps', true)) element.append(stamp);
      else element.style.gridTemplateColumns = '1fr';
      element.append(text);
      $('#transcript').append(element);
    }
    $('#result-count').textContent = `${rows.filter(row => !row.partial).length} 段已确认`;
    if (debug) {
      $('#comparison').replaceChildren();
      for (const [key, title] of [['realtime', '临时文字'], ['candidate', '候选结果'], ['final', '最终结果'], ['correction', '后续修正']]) {
        const row = document.createElement('div');
        row.className = 'compare-row';
        const label = document.createElement('span');
        label.textContent = title;
        const text = document.createElement('div');
        text.textContent = state.comparison[key] || '尚无结果';
        row.append(label, text);
        $('#comparison').append(row);
      }
    }
  }

  function readConfig() {
    const selected = state.publicConfig.models.find(model => model.id === $('#model').value);
    return {
      model: selected.id, input_mode: state.mode,
      language: { '中文': 'zh', 'English': 'en', '自动检测': null }[$('#language').value],
      segmentation: $('#segmentation').selectedIndex ? 'manual' : 'auto',
      realtime: state.mode !== 'file' && checked('#partials', true),
      realtime_mode: debug ? ($('#realtime-mode').selectedIndex ? 'native' : 'snapshot') : (selected.streaming ? 'native' : 'snapshot'),
      realtime_model: debug ? ($('#realtime-model').value || null) : null,
      interval: number('#interval', .2), stabilize: checked('#stabilize', true),
      word_timestamps: checked('#timestamps', true) && selected.word_timestamps,
      silence: number('#silence', .8), vad_sensitivity: number('#vad', .4),
      wake: checked('#wake') && !$('#segmentation').selectedIndex ? $('#keyword').value : null,
      wake_sensitivity: number('#wake-sensitivity', .5), wake_timeout: number('#wake-timeout', 5),
      candidates: checked('#candidates'), correction: checked('#correction'),
      max_audio_samples: number('#buffer', 480000), max_pending_units: number('#pending', 8),
      max_audio_seconds: number('#max-seconds', null),
      chunk_ms: number('#chunk', 100), pace: $('#pace')?.selectedIndex ? 'fast' : 'realtime',
    };
  }

  function syncCapabilities() {
    if (!state.publicConfig) return;
    const selected = state.publicConfig.models.find(model => model.id === $('#model').value);
    options($('#device'), [[selected.device, selected.device]]);
    $('#device').disabled = true;
    $('#compatibility').textContent = `${selected.engine} · ${selected.streaming ? '原生流式' : '快照实时'} · ${selected.word_timestamps ? '支持词时间戳' : '无词时间戳'}`;
    if (debug) {
      const realtime = state.publicConfig.models.find(model => model.id === $('#realtime-model').value) || selected;
      $('#realtime-mode').selectedIndex = realtime.streaming ? 1 : 0;
      $('#realtime-mode').disabled = true;
    }
    const manual = $('#segmentation').selectedIndex === 1;
    if (manual || !state.publicConfig.wakes.length) $('#wake').checked = false;
    $('#wake').disabled = manual || !state.publicConfig.wakes.length || busy() || state.phase === 'finished';
    $('#wake-options').hidden = !$('#wake').checked;
    $('#silence').disabled = manual || busy() || state.phase === 'finished';
    if (debug) {
      if (!$('#candidates').checked) $('#correction').checked = false;
      $('#correction').disabled = !$('#candidates').checked || busy() || state.phase === 'finished';
      const wake = state.publicConfig.wakes.find(item => item.id === $('#keyword').value);
      $('#wake-backend').value = wake?.backend === 'porcupine' ? 'Porcupine' : 'OpenWakeWord';
      $('#wake-backend').disabled = true;
    }
    $('#input-policy').textContent = manual ? '手动分段 · 不启用唤醒' : `自动分段 · 停顿 ${$('#silence').value} 秒${$('#wake').checked ? ' · 唤醒已启用' : ''}`;
    if (state.mode === 'file') $('#input-policy').textContent = '整段文件转写';
    $('#timer').hidden = state.mode === 'file';
    $('#model-summary').textContent = `${selected.id} · ${$('#language').value}`;
    updatePreview();
  }

  function updatePreview() {
    if (!debug) return;
    const config = readConfig();
    const selected = state.publicConfig.models.find(model => model.id === $('#model').value);
    $('#config-preview').textContent = JSON.stringify(config, null, 2);
    $('#code-note').textContent = '这是对应公开 API 的最小示例。模型类型和目录需填写本机值；完整配置映射见 apps/web/server/service.py。';
    $('#code-preview').textContent = config.input_mode === 'file'
      ? `from speechio import ModelConfig, Transcriber\n\nmodel = ModelConfig(engine="${selected.engine}", model_path="填写本地模型目录", device="${selected.device}")\nwith Transcriber(model, max_audio_seconds=${config.max_audio_seconds ?? 'None'}) as recognizer:\n    result = recognizer.transcribe_file("audio.wav")\n    print(result.text)`
      : `from speechio import ModelConfig, SessionConfig\nfrom speechio.runtime.asr import ASRRuntime\n${config.segmentation === 'auto' ? 'from speechio import VadConfig\nfrom speechio.bundled import default_vad_model\n' : ''}\nwith ASRRuntime(ModelConfig(engine="${selected.engine}", model_path="填写本地模型目录", device="${selected.device}")) as runtime:\n    session = runtime.${config.segmentation === 'auto' ? 'create_auto_session(vad=VadConfig(model_path=default_vad_model()))' : 'create_session(SessionConfig())'}\n${config.segmentation === 'manual' ? '    session.start()\n' : ''}    # 将 16 kHz 单声道 int16 音频持续送入 session.feed_audio(pcm)\n    # 停止后调用 ${config.segmentation === 'auto' ? 'session.end_input()' : 'session.end_turn()'}，继续读取 next_event()\n    # 消费完成事件后，调用 session.close()`;
  }

  function updateControls() {
    const active = busy();
    const connected = state.phase === 'finished';
    $('#start').disabled = !state.publicConfig || ['preparing', 'draining', 'stopping', 'closing'].includes(state.phase) || connected || (state.mode !== 'live' && !state.file && !active);
    $('#start-label').textContent = state.phase === 'recording' ? (state.mode === 'file' ? '识别中…' : '停止输入') : state.phase === 'preparing' ? '准备中…' : state.phase === 'draining' ? '处理剩余音频…' : state.mode === 'live' ? '开始录音' : '开始转写';
    if (state.mode === 'file' && active) $('#start').disabled = true;
    $('#end-turn').disabled = state.phase !== 'recording' || state.mode === 'file';
    for (const element of document.querySelectorAll('.field input, .field select, .toggle input, [data-mode], #file-picker, #microphone')) element.disabled = active || connected;
    document.body.classList.toggle('running', state.phase === 'recording');
    if (debug) {
      $('#prepare').disabled = active || connected;
      $('#release').disabled = active || connected;
      $('#cancel-turn').disabled = state.phase !== 'recording' || state.mode === 'file';
      $('#reset').disabled = !state.client || state.mode === 'file' || state.phase === 'preparing';
      $('#close-session').disabled = !state.client || state.phase === 'preparing';
      $('#candidate').disabled = state.phase !== 'recording' || !state.config?.candidates;
      $('#correct').disabled = !state.client || !state.config?.correction || !state.comparison.final;
      $('#session-state').textContent = state.phase;
    }
    syncCapabilities();
  }

  function phase(name, message) {
    state.phase = name;
    $('#status').textContent = message;
    updateControls();
  }

  async function abort(error) {
    if (state.phase === 'error') return;
    showError(error);
    phase('error', '运行失败，请检查提示后重试');
    const microphone = state.microphone;
    state.microphone = null;
    await microphone?.dispose();
    state.client?.close();
    state.client = null;
    updateControls();
  }

  function onMessage(message) {
    if (['metrics', 'input_state'].includes(message.type) && state.phase === 'recording' && message.wake?.enabled) {
      const labels = { awaiting_wake: '等待唤醒词', awaiting_speech: '已唤醒，等待说话', recording: '正在接收语音', followup: '等待下一句话', grace: '正在进入唤醒监听' };
      $('#status').textContent = labels[message.wake.state] || '正在监听';
    }
    log(message);
    if (message.type === 'error') {
      if (message.fatal) void abort(new Error(message.message));
      else showError(new Error(message.message));
      return;
    }
    if (message.type === 'metrics') {
      $('#metric-buffer').textContent = `${(message.capacity.task_audio_samples / 16000).toFixed(1)} 秒`;
      $('#metric-queue').textContent = message.capacity.tasks;
      $('#metric-elapsed').textContent = `${message.input_seconds.toFixed(1)} 秒`;
      $('#timer').textContent = clock(message.input_seconds);
      $('#model-state').textContent = `模型 ${message.model_state}`;
      if (message.wake) $('#session-state').textContent = `唤醒：${message.wake.state}`;
      return;
    }
    if (message.type === 'finished') {
      const status = message.success === false ? '处理已结束，部分识别失败，请查看错误' : '已完成，所有已接收音频处理完毕';
      state.completionMessage = status;
      phase(debug ? 'finished' : 'closing', status);
      if (!debug) state.client?.close();
      return;
    }
    if (message.type === 'ack' && message.command === 'reset') {
      state.completionMessage = null;
      state.rows.clear();
      state.comparison = {};
      renderResults();
      // 已结束的输入需要重新打开麦克风，因此重置后由新任务继续。
      if (state.phase === 'finished' || (!state.microphone && state.mode === 'live')) {
        state.client?.close();
        phase('closing', '会话已重置，正在结束旧连接');
      }
      else phase('recording', '会话已重置');
    }
    if (message.type === 'event') {
      if (message.kind === 'cancelled' && message.turn) {
        state.rows.delete(`${message.turn.generation}:${message.turn.turn_id}`);
        renderResults();
      }
      if (message.error) showError(new Error(`${message.error.code}：${message.error.message}`));
      if (message.result) {
        const key = message.turn ? `${message.turn.generation}:${message.turn.turn_id}` : 'file';
        const kind = message.kind;
        state.comparison[kind] = message.result.text;
        if (['realtime', 'final', 'correction'].includes(kind)) {
          const previous = state.rows.get(key);
          if (kind !== 'realtime' || !previous || previous.partial) {
            state.rows.set(key, {
              text: message.stabilization?.display_text ?? message.result.text,
              partial: kind === 'realtime',
              seconds: message.stream_start_sample == null ? (message.result.words?.[0]?.start_sample ?? null) : message.stream_start_sample,
            });
            const row = state.rows.get(key);
            if (row.seconds != null) row.seconds /= 16000;
          }
        }
        if (debug && kind === 'final' && message.completed_at_monotonic != null) {
          $('#metric-latency').textContent = `${(message.completed_at_monotonic - message.started_at_monotonic).toFixed(2)} 秒`;
        }
        renderResults();
        updateControls();
      }
    }
  }

  async function start() {
    $('#error').hidden = true;
    state.rows.clear();
    state.comparison = {};
    state.receivedSeconds = 0;
    state.completionMessage = null;
    renderResults();
    phase('preparing', '正在准备输入并加载本地模型');
    try {
      const config = readConfig();
      if (state.mode === 'live') {
        state.microphone = new Microphone();
        config.sample_rate = await state.microphone.open($('#microphone').value);
        const devices = await navigator.mediaDevices.enumerateDevices();
        options($('#microphone'), [['', '默认麦克风'], ...devices.filter(d => d.kind === 'audioinput').map(d => [d.deviceId, d.label || '麦克风'])], $('#microphone').value);
      } else {
        if (!state.file) throw new Error('请先选择音频文件');
        if (state.file.size > state.publicConfig.upload_limit_bytes) throw new Error('文件超过本机服务的上传大小限制');
        const upload = await request('/api/uploads', { method: 'POST', body: state.file });
        state.token = upload.token;
      }
      state.config = config;
      const client = new SpeechClient(onMessage, () => {
        if (state.client !== client) return;
        state.client = null;
        if (busy() && state.phase !== 'closing') void abort(new Error('服务连接中断；本次未完成的音频没有被确认'));
        else {
          if (state.phase === 'finished') phase('idle', '会话已关闭');
          if (state.phase === 'closing') phase('idle', state.completionMessage || '会话已关闭');
          updateControls();
        }
      });
      state.client = client;
      await client.start(config, state.token, debug);
      state.token = null;
      phase('recording', state.mode === 'live' ? '正在接收音频' : '正在识别文件');
      state.microphone?.start(bytes => {
        client.audio(bytes);
        state.receivedSeconds += bytes.byteLength / 2 / config.sample_rate;
        $('#timer').textContent = clock(state.receivedSeconds);
      }, error => void abort(error), level => $('.wave').style.setProperty('--level', level));
    } catch (error) {
      await abort(error);
      if (state.token) await request(`/api/uploads/${state.token}`, { method: 'DELETE' }).catch(() => {});
      state.token = null;
    }
  }

  async function stop() {
    phase('stopping', '正在送完麦克风尾部音频');
    try {
      await state.microphone?.stop();
      state.microphone = null;
      $('.wave').style.setProperty('--level', 0);
      state.client.command('end_input');
      phase('draining', '输入已停止，正在等待最终结果');
    } catch (error) { await abort(error); }
  }

  async function closeSession() {
    state.completionMessage = null;
    await state.microphone?.dispose();
    state.microphone = null;
    state.client?.close();
    phase(state.client ? 'closing' : 'idle', state.client ? '正在关闭会话，等待资源释放' : '已就绪，可以开始');
  }

  function chooseFile(file) {
    if (!file || busy()) return;
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
    state.file = file;
    state.fileUrl = URL.createObjectURL(file);
    $('#filename').textContent = file.name;
    $('#file-meta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · 处理后删除服务端临时文件`;
    $('#player').src = state.fileUrl;
    $('#player').hidden = false;
    updateControls();
  }

  function download(value, filename, type) {
    const url = URL.createObjectURL(new Blob([value], { type }));
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const fullText = () => [...state.rows.values()].filter(row => !row.partial).map(row => row.text).join('\n');
  const act = callback => async () => { try { await callback(); } catch (error) { showError(error); } };

  $('#start').onclick = act(() => state.phase === 'recording' ? stop() : start());
  $('#end-turn').onclick = act(() => state.client.command('end_turn'));
  $('#file-picker').onchange = event => chooseFile(event.target.files[0]);
  $('#drop-zone').ondragover = event => event.preventDefault();
  $('#drop-zone').ondrop = event => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]); };
  $('#copy').onclick = act(async () => { await navigator.clipboard.writeText(fullText()); toast('已复制确认文本'); });
  $('#export').onclick = () => download(fullText(), 'speechio.txt', 'text/plain;charset=utf-8');
  $('#clear').onclick = () => { state.rows.clear(); state.comparison = {}; renderResults(); };
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.onclick = () => {
      state.mode = button.dataset.mode;
      document.querySelectorAll('[data-mode]').forEach(tab => {
        tab.classList.toggle('active', tab === button);
        tab.setAttribute('aria-pressed', tab === button);
      });
      $('#microphone-input').hidden = state.mode !== 'live';
      $('#file-input').hidden = state.mode === 'live';
      document.querySelectorAll('[data-live-only]').forEach(element => { element.hidden = state.mode === 'file'; });
      document.querySelectorAll('[data-file-only]').forEach(element => { element.hidden = state.mode === 'live'; });
      updateControls();
    };
  });
  document.querySelectorAll('.field input, .field select, .toggle input').forEach(element => {
    element.addEventListener('change', () => { syncCapabilities(); renderResults(); });
  });
  $('#engine').onchange = () => {
    const models = state.publicConfig.models.filter(model => model.engine === $('#engine').value);
    options($('#model'), models.map(model => [model.id, model.id]));
    syncCapabilities();
  };

  if (debug) {
    $('#prepare').onclick = act(async () => {
      phase('preparing', '正在准备模型');
      try {
        const result = await request('/api/debug/prepare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(readConfig()) });
        $('#model-state').textContent = '模型已准备';
        log({ type: 'model.prepared', ...result });
      } finally { phase('idle', '准备操作已结束'); }
    });
    $('#release').onclick = act(async () => {
      phase('preparing', '正在释放模型');
      try {
        const result = await request('/api/debug/release', { method: 'POST' });
        $('#model-state').textContent = '模型已释放';
        log({ type: 'model.released', ...result });
      } finally { phase('idle', '释放操作已结束'); }
    });
    for (const [id, command] of [['cancel-turn', 'cancel_turn'], ['reset', 'reset'], ['candidate', 'candidate'], ['correct', 'correct']]) {
      $(`#${id}`).onclick = act(() => state.client.command(command));
    }
    $('#close-session').onclick = act(closeSession);
    $('#event-filter').onchange = renderEvents;
    $('#export-events').onclick = () => download(JSON.stringify(state.events, null, 2), 'speechio-events.json', 'application/json');
    $('#copy-config').onclick = act(async () => { await navigator.clipboard.writeText($('#config-preview').textContent); toast('已复制配置'); });
    for (const name of ['text', 'compare']) {
      $(`#${name}-tab`).onclick = () => {
        $('#transcript').hidden = name !== 'text';
        $('#comparison').hidden = name !== 'compare';
        for (const other of ['text', 'compare']) {
          $(`#${other}-tab`).classList.toggle('active', other === name);
          $(`#${other}-tab`).setAttribute('aria-pressed', other === name);
        }
      };
    }
  } else {
    const drawer = $('#drawer');
    const showDrawer = visible => {
      drawer.hidden = !visible;
      $('#backdrop').hidden = !visible;
      if (visible) $('#close-settings').focus();
      else $('#open-settings').focus();
    };
    $('#open-settings').onclick = () => showDrawer(true);
    for (const id of ['close-settings', 'done-settings', 'backdrop']) $(`#${id}`).onclick = () => showDrawer(false);
    document.addEventListener('keydown', event => {
      if (drawer.hidden) return;
      if (event.key === 'Escape') showDrawer(false);
      if (event.key === 'Tab') {
        const focusable = [...drawer.querySelectorAll('button, select, input')].filter(el => !el.disabled && el.getClientRects().length);
        const first = focusable[0], last = focusable.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    });
    $('#new-session').onclick = act(async () => {
      await closeSession();
      state.rows.clear();
      state.comparison = {};
      renderResults();
    });
  }

  window.addEventListener('pagehide', () => {
    void state.microphone?.dispose();
    state.client?.socket?.close();
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  });
  renderResults();
  updateControls();
  try {
    state.publicConfig = await request('/api/config');
    const models = state.publicConfig.models;
    options($('#engine'), [...new Set(models.map(model => model.engine))].map(name => [name, name]));
    options($('#model'), models.filter(model => model.engine === $('#engine').value).map(model => [model.id, model.id]));
    options($('#keyword'), state.publicConfig.wakes.map(wake => [wake.id, wake.id]));
    if (debug) options($('#realtime-model'), [['', '共用最终模型'], ...models.map(model => [model.id, model.id])]);
    if (debug) $('#model-state').textContent = `模型 ${state.publicConfig.model_state}`;
    $('#other-page').hidden = !debug && !state.publicConfig.debug;
    phase('idle', '已就绪，可以开始');
    log({ type: 'connected', message: '已读取本机服务配置' });
  } catch (error) { showError(error); }
}
