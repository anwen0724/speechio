/** 共享页面行为。B 保持简洁；C 额外订阅事件、容量与模型控制。 */
import { mount } from './ui.js';
import { SpeechClient, request } from './client.js';
import { Microphone } from './audio.js';
import { mountFeatures } from './features.js';

export async function startPage(debug) {
  mount(debug);
  const $ = selector => document.querySelector(selector);
  const state = {
    mode: 'live', phase: 'idle', client: null, microphone: null, file: null,
    fileUrl: null, token: null, rows: new Map(), events: [], comparison: {},
    receivedSeconds: 0, config: null, publicConfig: null, completionMessage: null,
    waitTimer: null,
  };
  const remote = model => model?.source === 'remote';
  const backend = model => `${model.source}:${remote(model) ? model.provider : model.engine}`;
  const backendLabel = model => remote(model)
    ? `${model.provider === 'bailian' ? '百炼' : model.provider}（云端）`
    : `${{ whisper: 'Whisper', nemotron: 'Nemotron' }[model.engine] || model.engine}（本地）`;
  const selectedModel = (prefix = '') => state.publicConfig.models.find(model => model.id === $(`#${prefix}model`).value);
  const busy = () => ['preparing', 'recording', 'draining', 'stopping', 'closing'].includes(state.phase);
  const number = (id, fallback) => $(id)?.value === '' ? fallback : Number($(id)?.value ?? fallback);
  const checked = (id, fallback = false) => $(id)?.checked ?? fallback;
  const clock = seconds => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;

  function options(element, entries, preferred) {
    element.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
    if (entries.some(([value]) => value === preferred)) element.value = preferred;
  }

  // 主模型按输入能力筛选；后端和模型都保留仍然可用的当前选择。
  function refreshModels() {
    const models = state.publicConfig.models.filter(model => state.mode === 'file'
      ? model.transcription || model.file_transcription : model.transcription);
    options($('#engine'), [...new Map(models.map(model => [backend(model), backendLabel(model)]))], $('#engine').value);
    options($('#model'), models.filter(model => backend(model) === $('#engine').value)
      .map(model => [model.id, model.id]), $('#model').value);
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

  // 只在模型变化时重置默认值，其他控件变化不能覆盖用户刚选的设备和精度。
  function syncModelControls(model, prefix = '') {
    const device = $(`#${prefix}device`);
    const changed = device.dataset.model !== model.id;
    const locked = busy() || state.phase === 'finished';
    const cloud = remote(model);
    if (!prefix) {
      $('#file-task-runtime').hidden = state.mode !== 'file' || !model.file_transcription;
      const taskTimeout = $('#task-timeout');
      taskTimeout.min = '0.01';
      taskTimeout.placeholder = '填写正数';
      taskTimeout.disabled = locked;
      if (changed && model.file_transcription) taskTimeout.value = model.task_timeout_seconds;
    }
    device.closest('.field').hidden = cloud;
    if (debug) {
      $(`#${prefix}runtime-fields`).hidden = cloud;
      $(`#${prefix}remote-runtime`).hidden = !cloud;
      const timeout = $(`#${prefix}timeout`);
      timeout.min = '0.01';
      timeout.placeholder = '填写正数';
      timeout.disabled = locked;
      if (changed && cloud) timeout.value = model.timeout_seconds;
    }
    if (changed && !cloud) {
      options(device, model.devices.map(value => [value, value.toUpperCase()]), model.device);
      if (debug) {
        options($(`#${prefix}compute-type`), [
          ['inherit', `配置默认（${model.compute_type ?? '引擎默认'}）`],
          ['default', '引擎默认'],
          ...model.compute_types.filter(value => value !== null).map(value => [value, value]),
        ], 'inherit');
        $(`#${prefix}device-index`).value = model.device_index;
      }
    }
    device.dataset.model = model.id;
    device.disabled = locked || cloud || model.devices.length === 1;
    if (debug) {
      const index = $(`#${prefix}device-index`);
      index.min = '0';
      index.step = '1';
      if (device.value === 'cpu') index.value = '0';
      index.disabled = locked || cloud || device.value === 'cpu';
      $(`#${prefix}compute-type`).disabled = locked || cloud;
    }
  }

  function modelOverrides(prefix = '', fieldPrefix = '') {
    const model = selectedModel(prefix);
    if (remote(model)) {
      return {
        ...(debug ? { [`${fieldPrefix}timeout_seconds`]: number(`#${prefix}timeout`, null) } : {}),
        ...(!prefix && state.mode === 'file' && model.file_transcription
          ? { task_timeout_seconds: number('#task-timeout', null) } : {}),
      };
    }
    const result = { [`${fieldPrefix}device`]: $(`#${prefix}device`).value };
    if (debug) {
      result[`${fieldPrefix}device_index`] = Number($(`#${prefix}device-index`).value);
      const precision = $(`#${prefix}compute-type`).value;
      // 不发送表示继承；明确发送 null 表示取消文件中指定的精度。
      if (precision !== 'inherit') result[`${fieldPrefix}compute_type`] = precision === 'default' ? null : precision;
    }
    return result;
  }

  function showEffective(config) {
    if (!config) return;
    const main = config.completion_source === 'streaming' ? config.realtime : config.main;
    $('#model-summary').textContent = remote(main)
      ? `${main.id} · 云端 · ${main.task_timeout_seconds == null ? `请求超时 ${main.timeout_seconds}` : `任务最多等待 ${main.task_timeout_seconds}`} 秒`
      : `${main.id} · ${main.device.toUpperCase()} · ${main.compute_type ?? '引擎默认'}`;
    if (debug) $('#effective-config').textContent = JSON.stringify(config, null, 2);
  }

  function readConfig() {
    const selected = state.publicConfig.models.find(model => model.id === $('#model').value);
    return {
      model: selected.id, input_mode: state.mode,
      ...modelOverrides(),
      ...(debug && state.mode !== 'file' && checked('#partials', true) && $('#realtime-model').value
        ? modelOverrides('realtime-', 'realtime_') : {}),
      language: { '中文': 'zh', 'English': 'en', '自动检测': null }[$('#language').value],
      segmentation: $('#segmentation').selectedIndex ? 'manual' : 'auto',
      realtime: state.mode !== 'file' && checked('#partials', true),
      completion_mode: debug && state.mode !== 'file' && checked('#retranscribe') ? 'final' : 'auto',
      realtime_mode: debug ? ($('#realtime-mode').value === 'streaming' ? 'streaming' : 'snapshot') : (selected.streaming ? 'streaming' : 'snapshot'),
      realtime_model: debug && state.mode !== 'file' && checked('#partials', true) ? ($('#realtime-model').value || null) : null,
      interval: number('#interval', .2), stabilize: checked('#stabilize', true),
      word_timestamps: checked('#timestamps', true) && selected.word_timestamps,
      silence: number('#silence', .8), vad_sensitivity: number('#vad', .4),
      wake: state.mode !== 'file' && checked('#wake') && !$('#segmentation').selectedIndex ? $('#keyword').value : null,
      wake_sensitivity: number('#wake-sensitivity', .5), wake_timeout: number('#wake-timeout', 5),
      candidates: state.mode !== 'file' && checked('#candidates'), correction: state.mode !== 'file' && checked('#correction'),
      max_audio_samples: number('#buffer', 480000), max_pending_units: number('#pending', 8),
      max_audio_seconds: number('#max-seconds', null),
      chunk_ms: number('#chunk', 100), pace: $('#pace')?.selectedIndex ? 'fast' : 'realtime',
    };
  }

  function syncCapabilities() {
    if (!state.publicConfig) return;
    const selected = state.publicConfig.models.find(model => model.id === $('#model').value);
    if (!selected) {
      $('#compatibility').textContent = '当前输入方式没有可用模型，请切换输入方式或配置模型后重启服务。';
      $('#model-summary').textContent = '没有可用模型';
      $('#file-task-runtime').hidden = true;
      $('#remote-note').hidden = true;
      if (debug) {
        $('#config-preview').textContent = '';
        $('#code-preview').textContent = '';
      }
      return;
    }
    syncModelControls(selected);
    const realtime = (debug && selectedModel('realtime-')) || selected;
    const canPreview = realtime.snapshot || realtime.streaming;
    const choice = `${selected.id}:${realtime.id}`;
    const partials = $('#partials');
    // 切换模型时按能力启用预览；语言等其他设置不能覆盖用户手动关闭的选择。
    if (partials.dataset.model !== choice) partials.checked = canPreview;
    partials.dataset.model = choice;
    if (!canPreview) partials.checked = false;
    partials.disabled = !canPreview || state.mode === 'file' || busy() || state.phase === 'finished';
    const streaming = state.mode !== 'file' && partials.checked && realtime.streaming;
    const reuseStream = streaming && !(debug && checked('#retranscribe'));
    const capability = selected.file_transcription ? '长音频文件识别' : selected.streaming ? '原生流式' : selected.snapshot ? '快照实时' : '仅整段识别';
    $('#compatibility').textContent = `${backendLabel(selected)} · ${capability} · ${selected.word_timestamps ? '词时间戳以返回结果为准' : '无词时间戳'}${remote(selected) && !selected.api_key_configured ? ' · 密钥未配置，请填写 config.toml 的 apikey 或 apps/web/.env 后重启服务' : ''}`;
    if (debug) {
      options($('#realtime-mode'), realtime.streaming ? [['streaming', '原生流式']]
        : realtime.snapshot ? [['snapshot', '快照识别']] : [['none', '所选模型不支持实时预览']]);
      $('#realtime-mode').disabled = true;
      $('#retranscribe').disabled = !streaming || busy() || state.phase === 'finished';
      $('#completion-note').textContent = reuseStream
        ? `最终结果来自 ${realtime.id} 的流式任务，不再调用主模型识别整段音频。`
        : streaming
          ? '流式结束后，使用主模型重新识别整段音频；云端可能增加一次识别费用。'
          : '当前没有使用流式识别，最终结果由主模型识别整段音频得到。';
      const independent = state.mode !== 'file' && partials.checked && Boolean($('#realtime-model').value);
      $('#realtime-runtime').hidden = !independent;
      // 即使主模型不支持预览，也允许选择另一个有流式能力的模型。
      $('#realtime-model').disabled = busy() || state.phase === 'finished' || state.mode === 'file';
      $('#realtime-runtime-note').textContent = independent
        ? `独立实时模型使用下方${remote(realtime) ? '请求超时' : '设备与精度'}。候选和修正仍共用主模型。`
        : '启用实时识别时共用主模型；也可选择独立实时模型。';
      if (independent) syncModelControls(realtime, 'realtime-');
    }
    const cloudPreview = state.mode !== 'file' && partials.checked && remote(realtime);
    const cloud = remote(selected) || cloudPreview;
    if (!debug) $('#recognition-location').textContent = cloud ? '云端识别' : '本地识别';
    $('#remote-note').hidden = !cloud;
    const notes = [];
    if (cloud) notes.push('音频将由本机服务发送至百炼识别。');
    if (remote(selected) && !selected.streaming && !selected.file_transcription) notes.push('短音频模型最多 300 秒，还受 HTTP 编码后大小限制；超限不会自动切分。');
    if (selected.file_transcription) notes.push('长音频文件模型支持最长 12 小时、最大 1 GiB 的文件；还受本机上传上限约束。提交后等待最终结果，不逐字显示。关闭页面不会撤销已经提交的云端任务。');
    if (cloudPreview && reuseStream) notes.push('最终文字直接采用流式任务的最终结果，不会自动再次识别整段音频。');
    if (streaming && !reuseStream && remote(selected)) notes.push('已开启重识别：结束后主模型将额外识别整段音频，可能增加费用。');
    if (debug && remote(selected) && state.mode !== 'file') notes.push('主动请求候选或修正会增加识别请求。');
    if (cloudPreview && !realtime.api_key_configured) notes.push('实时模型密钥未配置，请填写 config.toml 的 apikey 或 apps/web/.env 后重启服务。');
    $('#remote-note').textContent = notes.join(' ');
    $('#audio-location').textContent = `${debug ? '本机单任务' : 'SpeechIO 示例应用'} · ${cloud ? '音频发送至百炼识别' : '音频由本机服务处理'}`;
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
    $('#timer').hidden = false;
    $('#timer').title = state.mode === 'file' ? '本次文件上传与识别已用时间' : '已输入音频时长';
    $('#model-summary').textContent = `${selected.id} · ${$('#language').value}`;
    updatePreview();
  }

  function updatePreview() {
    if (!debug) return;
    const config = readConfig();
    const selected = state.publicConfig.models.find(model => model.id === $('#model').value);
    $('#config-preview').textContent = JSON.stringify(config, null, 2);
    $('#code-note').textContent = '以下使用核心包公开 API；本地目录和环境变量由使用者填写，未包含服务端密钥或地址。';
    const pythonValue = value => value === null ? 'None' : JSON.stringify(value);
    const modelCode = (model, prefix = '') => {
      if (remote(model)) {
        const timeout = config[`${prefix}timeout_seconds`] ?? model.timeout_seconds;
        return `RemoteModelConfig(provider=${pythonValue(model.provider)}, model=${pythonValue(model.model)}, api_key=os.environ["DASHSCOPE_API_KEY"], timeout_seconds=${timeout})`;
      }
      const precisionKey = `${prefix}compute_type`;
      const precision = Object.hasOwn(config, precisionKey) ? config[precisionKey] : model.compute_type;
      const device = config[`${prefix}device`] ?? model.device;
      const index = device === 'cpu' ? 0 : (config[`${prefix}device_index`] ?? model.device_index);
      return `LocalModelConfig(adapter=${pythonValue(model.engine)}, model_path="填写本地模型目录", device=${pythonValue(device)}, device_index=${index}, compute_type=${pythonValue(precision)})`;
    };
    const lines = ['import os', 'from speechio import LocalModelConfig, RemoteModelConfig, Transcriber, TranscriptionOptions, RealtimeConfig, SessionConfig', '', `model = ${modelCode(selected)}`];
    if (config.input_mode === 'file') {
      lines.push(`with Transcriber(model, max_audio_seconds=${config.max_audio_seconds ?? 'None'}) as transcriber:`,
        `    result = transcriber.transcribe_file("audio.wav", options=TranscriptionOptions(language=${pythonValue(config.language)}), task_timeout_seconds=${config.task_timeout_seconds ?? 3600})`, '    print(result.text)');
    } else {
      lines.push('from speechio import STTRuntime');
      if (config.realtime && config.realtime_model) {
        const realtime = state.publicConfig.models.find(model => model.id === config.realtime_model);
        lines.push(`realtime_model = ${modelCode(realtime, 'realtime_')}`);
      }
      const realtime = config.realtime
        ? `RealtimeConfig(mode=${pythonValue(config.realtime_mode)}, interval_seconds=${config.interval}${config.realtime_model ? ', config=realtime_model' : ''})` : 'None';
      lines.push(`with STTRuntime(model, realtime=${realtime}) as runtime:`);
      if (config.segmentation === 'auto') {
        lines.push('    from speechio import VadConfig', '    from speechio.bundled import default_vad_model',
          `    session = runtime.create_auto_session(config=SessionConfig(options=TranscriptionOptions(language=${pythonValue(config.language)}), realtime=${config.realtime ? 'True' : 'False'}, completion_mode=${pythonValue(config.completion_mode)}), vad=VadConfig(model_path=default_vad_model()))`);
      } else {
        lines.push(`    session = runtime.create_session(SessionConfig(options=TranscriptionOptions(language=${pythonValue(config.language)}), realtime=${config.realtime ? 'True' : 'False'}, completion_mode=${pythonValue(config.completion_mode)}))`, '    session.start()');
      }
      lines.push('    # 输入 16 kHz 单声道 int16 音频：session.feed_audio(pcm)',
        `    # 输入结束：session.${config.segmentation === 'auto' ? 'end_input' : 'end_turn'}()`,
        '    # 继续读取 next_event() 直到完成，最后 session.close()');
    }
    $('#code-preview').textContent = lines.join('\n');
  }

  function updateControls() {
    const active = busy();
    const connected = state.phase === 'finished';
    const hasModel = state.publicConfig && selectedModel();
    $('#start').disabled = !hasModel || ['preparing', 'draining', 'stopping', 'closing'].includes(state.phase) || connected || (state.mode !== 'live' && !state.file && !active);
    $('#start-label').textContent = state.phase === 'recording' ? (state.mode === 'file' ? '识别中…' : '停止输入') : state.phase === 'preparing' ? '准备中…' : state.phase === 'draining' ? '处理剩余音频…' : state.mode === 'live' ? '开始录音' : '开始转写';
    if (state.mode === 'file' && active) $('#start').disabled = true;
    $('#end-turn').disabled = state.phase !== 'recording' || state.mode === 'file';
    for (const element of (document.querySelector('#asr-content') || document).querySelectorAll('.field input, .field select, .toggle input, [data-mode], #file-picker, #microphone')) element.disabled = active || connected;
    document.body.classList.toggle('running', state.phase === 'recording');
    if (debug) {
      $('#prepare').disabled = !hasModel || active || connected;
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
    if (['idle', 'error', 'finished', 'closing'].includes(name)) {
      clearInterval(state.waitTimer);
      state.waitTimer = null;
    }
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
    if (message.type === 'ready') { state.lastEffective = message.effective_config; showEffective(message.effective_config); }
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
      $('#model-state').textContent = `识别资源 ${message.model_state}`;
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
    $('#timer').textContent = '00:00';
    if (state.mode === 'file') {
      const started = performance.now();
      state.waitTimer = setInterval(() => {
        $('#timer').textContent = clock((performance.now() - started) / 1000);
      }, 1000);
    }
    renderResults();
    phase('preparing', '正在准备输入与识别配置');
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
        phase('preparing', '正在上传文件到本机服务');
        const extension = state.file.name.match(/\.[a-z0-9]{1,10}$/i)?.[0].toLowerCase() ?? '.audio';
        const upload = await request('/api/uploads', {
          method: 'POST', headers: { 'X-Audio-Extension': extension }, body: state.file,
        });
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
      phase('recording', state.mode === 'live' ? '正在接收音频' : selectedModel().file_transcription
        ? '正在处理云端文件任务，等待最终结果（计时为已用时间）' : '正在识别文件');
      showEffective(state.lastEffective);
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
      document.querySelectorAll('[data-chunks-only]').forEach(element => { element.hidden = state.mode !== 'chunks'; });
      refreshModels();
      updateControls();
    };
  });
  document.querySelectorAll('.field input, .field select, .toggle input').forEach(element => {
    element.addEventListener('change', () => { syncCapabilities(); renderResults(); });
  });
  $('#engine').onchange = () => {
    refreshModels();
    updateControls();
  };

  if (debug) {
    $('#prepare').onclick = act(async () => {
      phase('preparing', '正在准备模型');
      try {
        const result = await request('/api/debug/prepare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(readConfig()) });
        const used = [result.effective_config.main, result.effective_config.realtime].filter(Boolean);
        $('#model-state').textContent = used.some(remote)
          ? '配置已就绪；云端将在识别时连接并验证权限' : '模型已准备';
        log({ type: 'model.prepared', ...result });
        state.lastEffective = result.effective_config;
      } finally { phase('idle', '准备操作已结束'); showEffective(state.lastEffective); }
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
    clearInterval(state.waitTimer);
    void state.microphone?.dispose();
    state.client?.socket?.close();
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  });
  renderResults();
  updateControls();
  try {
    state.publicConfig = await request('/api/config');
    const models = state.publicConfig.models;
    refreshModels();
    $('#upload-limit').textContent = `本机上传上限：${state.publicConfig.upload_limit_bytes / 1024 / 1024} MiB（1024 MiB = 1 GiB）`;
    options($('#keyword'), state.publicConfig.wakes.map(wake => [wake.id, wake.id]));
    if (debug) options($('#realtime-model'), [['', '共用主模型'], ...models.filter(model => model.streaming || model.snapshot).map(model => [model.id, `${model.id} · ${backendLabel(model)}`])]);
    if (debug) $('#model-state').textContent = `模型 ${state.publicConfig.model_state}`;
    if (!models.some(model => model.transcription)) $('[data-mode="file"]').click();
    $('#other-page').hidden = !debug && !state.publicConfig.debug;
    phase('idle', '已就绪，可以开始');
    log({ type: 'connected', message: '已读取本机服务配置' });
    mountFeatures(debug, state.publicConfig, busy, async () => {
      if (state.phase !== 'finished' || !state.client?.socket) return;
      const socket = state.client.socket;
      const closed = new Promise(resolve => socket.addEventListener('close', resolve, { once: true }));
      await closeSession();
      await closed;
    });
  } catch (error) { showError(error); }
}
