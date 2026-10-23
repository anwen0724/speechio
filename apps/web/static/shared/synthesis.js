/** 合成页面的控件和任务状态；模型调用始终由本机服务完成。 */
import { request } from './client.js';
import { AudioPlayback } from './playback.js';

const field = (name, title, input) => `<label class="field"><span class="field-title">${title}</span>${input.replace('NAME', `tts-${name}`)}</label>`;
const select = (name, title, options = '') => field(name, title, `<select id="NAME">${options}</select>`);
const input = (name, title, attrs = '') => field(name, title, `<input id="NAME" ${attrs}>`);

function layout(debug) {
  const settings = `${select('model', '合成模型')}
    ${input('voice', '音色 ID', 'list="tts-voices" placeholder="选择或输入音色 ID"')}<datalist id="tts-voices"></datalist>
    <p class="help">音色建议来自配置，也可直接填写模型支持的其他音色 ID。</p>
    ${select('language', '合成语言', '<option value="">模型默认</option><option value="zh">中文</option><option value="en">English</option>')}
    ${input('speed', '语速', 'type="number" min="0.5" max="2" step="0.1" value="1"')}
    <details class="settings-group" ${debug ? 'open' : ''}><summary>更多设置</summary>
      ${input('instruction', '语气与风格', 'placeholder="留空使用模型默认"')}
      ${select('sample-rate', '采样率')}
      ${debug ? `${input('timeout', '网络等待时间（秒）', 'type="number" min="0.1" step="any" value="30"')}
        ${input('buffer', '核心音频缓冲上限（秒）', 'type="number" min="0.1" step="any" value="10"')}
        <p class="help">缓冲是核心包尚未交付的音频量，超限会报错；不是录音时长限制。</p>` : ''}
    </details><p class="help" id="tts-model-note"></p>`;
  const modes = debug
    ? '<option value="full">完整合成</option><option value="stream" selected>音频流</option><option value="incremental">增量文本</option>'
    : '<option value="full">完整生成</option><option value="stream" selected>边生成边播放</option>';
  const content = `<div class="toolbar">${select('mode', '生成方式', modes)}
      ${!debug ? '<button class="quiet" id="tts-settings-open">合成设置</button>' : ''}</div>
    <section class="input-card tts-input">
      <label class="field" for="tts-text"><span class="field-title" id="tts-text-title">输入需要朗读的文字</span></label>
      <textarea id="tts-text" rows="7" maxlength="200000" placeholder="在这里输入或粘贴文字，让 SpeechIO 读给你听。"></textarea>
      <div class="toolbar"><span class="subtle" id="tts-count">0 字</span><button class="quiet" id="tts-clear">清空</button></div>
      <div class="tts-actions"><button class="primary" id="tts-start">生成语音</button>
        <button class="quiet" id="tts-send" hidden>发送这段文字</button><button class="quiet" id="tts-finish" hidden>结束文本输入</button>
        <button class="quiet" id="tts-cancel" disabled>取消生成</button></div>
      <p class="help" id="tts-mode-note"></p>
      <p class="tts-status" role="status" id="tts-status">请选择模型并输入文字</p>
      <div class="alert" id="tts-error" role="alert" hidden></div>
    </section>
    ${debug ? '<section class="metrics tts-metrics" id="tts-metrics"></section>' : ''}
    <section class="tts-result"><div class="toolbar"><h2>生成的语音</h2><button class="quiet" id="tts-stop">停止试听</button></div>
      <p class="help" id="tts-play-state">尚未播放</p><audio id="tts-player" controls hidden></audio>
      <div class="toolbar"><p class="help" id="tts-result-info">完成后可重播、拖动进度和下载 WAV。</p>
      <a class="quiet" id="tts-download" download="speechio.wav" hidden>下载 WAV</a></div>
    </section>
    ${debug ? `<section class="console"><div class="section-head"><h2>合成事件</h2><button class="quiet" id="tts-export">导出 JSON</button></div><div class="event-list" id="tts-events"></div></section>
      <details class="developer"><summary>配置与 Python 调用</summary><div class="developer-body">
        <p class="help">本次请求参数</p><pre id="tts-config"></pre><p class="help">服务端实际采用的参数</p><pre id="tts-effective">尚未开始</pre>
        <p class="help">Python 示例中的密钥从环境变量读取，不显示服务端密钥。</p><pre id="tts-code"></pre>
      </div></details>` : ''}`;
  return `<main class="${debug ? 'studio-main' : 'example-main'}">
    <div class="${debug ? 'studio-title' : 'hero'}"><div><p class="eyebrow">TEXT TO SPEECH</p><h1>${debug ? '语音合成调试台' : '写下来，听见它。'}</h1><p class="help">${debug ? '观察文本输入、音频返回与资源释放。' : '把文字变成声音，试听，再带走。'}</p></div>
      ${debug ? '<div class="session-actions"><button class="quiet" id="tts-prepare">准备模型</button><button class="quiet" id="tts-release">释放模型</button></div>' : ''}</div>
    ${debug ? `<div class="studio-grid"><aside class="config-panel"><h2>合成设置</h2><div class="tts-settings">${settings}</div></aside><div class="studio-content">${content}</div></div>` : content}
    <footer class="page-footer">SpeechIO · 结果在本机临时保留 30 分钟；累计最多 128 MiB，超额时先移除旧结果。</footer>
    </main>${!debug ? `<div class="backdrop" id="tts-backdrop" hidden></div><section class="drawer" id="tts-drawer" role="dialog" aria-modal="true" aria-label="合成设置" hidden>
      <div class="drawer-head"><h2>按你的方式朗读</h2><button id="tts-settings-close" aria-label="关闭合成设置">×</button></div>
      <p class="drawer-intro">配置作为默认值；这里的修改只用于本次页面请求。</p>${settings}
      <div class="drawer-foot"><span class="subtle">配置保留在当前页面</span><button class="primary" id="tts-settings-done">完成</button></div></section>` : ''}`;
}

export function mountSynthesis(root, debug, models) {
  root.innerHTML = layout(debug);
  const $ = id => root.querySelector(`#tts-${id}`);
  let active = false, ready = false, ended = false, socket = null, events = [], playback;
  let terminal = false;
  let cancelRequested = false;
  const model = () => models.find(item => item.id === $('model').value);
  const mode = () => $('mode').value;
  const status = message => { $('status').textContent = message; };
  const error = value => {
    $('error').hidden = false;
    $('error').textContent = value.message || String(value);
  };
  playback = new AudioPlayback(message => { $('play-state').textContent = message; });
  const action = fn => async () => {
    try {
      await fn();
    } catch (exc) {
      error(exc);
    }
  };

  function config() {
    return {
      model: $('model').value,
      mode: mode(),
      voice: $('voice').value,
      language: $('language').value || null,
      speed: Number($('speed').value),
      instruction: $('instruction').value || null,
      sample_rate: Number($('sample-rate').value) || null,
      ...(debug ? {
        timeout_seconds: Number($('timeout').value),
        max_buffer_seconds: Number($('buffer').value),
      } : {}),
    };
  }

  function code() {
    if (!debug) return;
    const c = config(), m = model();
    $('config').textContent = JSON.stringify(c, null, 2);
    const py = value => value === null ? 'None' : JSON.stringify(value);
    const options = ['voice', 'language', 'speed', 'instruction', 'sample_rate']
      .map(key => `    ${key}=${py(c[key])},`).join('\n');
    const calls = {
      full: `    audio = synth.synthesize(text, options=options)
    audio.save("speech.wav")`,
      stream: `    with synth.stream(text, options=options) as stream:
        for chunk in stream:
            print(len(chunk.data))  # 在应用中播放或保存 PCM 音频`,
      incremental: `    with synth.open_stream(options=options) as stream:
        # 单独读取音频，避免等待下一段文本时积压音频。
        def receive_audio():
            for chunk in stream:
                print(len(chunk.data))

        reader = Thread(target=receive_audio)
        reader.start()
        try:
            stream.send_text("你好，")
            stream.send_text("欢迎使用 SpeechIO。")
            stream.finish_input()
            reader.join()
        finally:
            stream.close()
            reader.join()`,
    };
    $('code').textContent = `import os
${c.mode === 'incremental' ? 'from threading import Thread\n' : ''}from speechio import RemoteModelConfig, SynthesisOptions, Synthesizer

config = RemoteModelConfig(
    provider=${py(m?.provider || 'bailian')},
    model=${py(m?.model || '')},
    api_key=os.environ["DASHSCOPE_API_KEY"],
    timeout_seconds=${c.timeout_seconds},
)
options = SynthesisOptions(
${options}
)
text = "你好，欢迎使用 SpeechIO。"
with Synthesizer(config, max_buffer_seconds=${c.max_buffer_seconds}) as synth:
${calls[c.mode]}`;
  }

  function controls() {
    const incremental = mode() === 'incremental';
    root.querySelectorAll('.field input, .field select').forEach(el => { el.disabled = active; });
    $('text').disabled = active && (!incremental || ended);
    $('clear').disabled = $('text').disabled;
    $('start').disabled = active || !model() || (!incremental && !$('text').value.trim());
    $('start').textContent = incremental ? '打开合成任务' : '生成语音';
    $('cancel').disabled = !active || terminal || cancelRequested;
    $('send').hidden = $('finish').hidden = !incremental;
    $('send').disabled = !ready || ended || !$('text').value.trim();
    $('finish').disabled = !ready || ended;
    if (debug) {
      $('prepare').disabled = active || !model();
      $('release').disabled = active;
    }
    $('count').textContent = `${$('text').value.length} 字`;
    $('mode-note').textContent = incremental
      ? '先打开任务，再逐段输入并发送；全部发送后点击“结束文本输入”，继续接收剩余音频。不要长时间停顿，服务可能关闭空闲连接。'
      : mode() === 'full' ? '完整生成返回后再试听。取消时等待正在执行的核心调用退出。'
      : '提交完整文字，收到音频即试听；停止试听不取消生成，取消生成会同时停止试听。';
    code();
  }

  function defaults() {
    const m = model();
    if (!m) {
      $('model-note').textContent = '尚未配置合成模型，请在 config.toml 添加 task="synthesis" 的模型条目后重启。';
      controls();
      return;
    }
    for (const name of ['voice', 'language', 'speed', 'instruction']) $(name).value = m.defaults[name] ?? '';
    $('voices').replaceChildren(...m.voices.map(voice => new Option(voice, voice)));
    $('sample-rate').replaceChildren(new Option(`模型默认（${m.capabilities.default_sample_rate} Hz）`, ''),
      ...(m.capabilities.sample_rates || [m.capabilities.default_sample_rate]).map(rate => new Option(`${rate} Hz`, rate)));
    $('sample-rate').value = m.defaults.sample_rate ?? '';
    if (debug) {
      $('timeout').value = m.timeout_seconds;
      $('buffer').value = m.max_buffer_seconds;
    }
    const cap = m.capabilities;
    for (const [id, allowed] of [
      ['language', cap.language_selection], ['speed', cap.speed_control],
      ['instruction', cap.instruction_control],
    ]) $(id).closest('.field').hidden = !allowed;
    const capabilities = { full: 'synthesis', stream: 'audio_streaming', incremental: 'text_streaming' };
    [...$('mode').options].forEach(option => {
      option.disabled = !cap[capabilities[option.value]];
    });
    if ($('mode').selectedOptions[0].disabled) {
      $('mode').value = [...$('mode').options].find(option => !option.disabled)?.value || '';
    }
    $('model-note').textContent = m.api_key_configured ? '密钥已配置。语速与音色是否适用，由所选模型检查。' : '密钥未配置：请填写 config.toml 的 apikey 或 apps/web/.env，保存后重启服务。';
    controls();
  }

  function log(event) {
    if (!debug) return;
    events.push({ time: new Date().toISOString(), ...event });
    if (events.length > 200) events.shift();
    const row = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = `${new Date().toLocaleTimeString()} · ${event.type}${event.bytes ? ` · ${event.bytes} bytes` : ''}`;
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(event, null, 2);
    row.append(summary, pre);
    $('events').append(row);
    if ($('events').children.length > 200) $('events').firstElementChild.remove();
    $('events').scrollTop = $('events').scrollHeight;
  }

  function metrics(value) {
    if (!debug) return;
    const rate = Number($('sample-rate').value) || model().capabilities.default_sample_rate;
    const duration = value.duration_seconds ?? value.bytes / (rate * 2);
    $('metrics').replaceChildren(...[
      ['首块音频等待', value.first_audio_seconds == null ? '—' : `${value.first_audio_seconds.toFixed(2)} 秒`],
      ['生成总耗时', value.elapsed_seconds == null ? '—' : `${value.elapsed_seconds.toFixed(2)} 秒`],
      ['音频时长', `${duration.toFixed(2)} 秒`],
      ['音频块 / 字节', `${value.chunks} / ${value.bytes}`],
    ].map(([title, text]) => {
      const div = document.createElement('div');
      div.className = 'metric';
      const label = document.createElement('span');
      label.textContent = title;
      const strong = document.createElement('strong');
      strong.textContent = text;
      div.append(label, strong);
      return div;
    }));
  }

  function send(type, text) {
    if (socket?.readyState !== WebSocket.OPEN) throw new Error('合成连接尚未就绪');
    socket.send(JSON.stringify({ type, ...(text === undefined ? {} : { text }) }));
  }

  async function start() {
    $('error').hidden = true;
    active = true;
    ready = ended = terminal = cancelRequested = false;
    controls();
    $('player').pause();
    $('player').removeAttribute('src');
    $('player').hidden = true;
    $('download').hidden = true;
    $('result-info').textContent = '正在生成；结果尚不可下载。';
    if (debug) {
      events = [];
      $('events').replaceChildren();
      metrics({ chunks: 0, bytes: 0 });
    }
    try {
      if (mode() !== 'full') {
        try {
          await playback.start(Number($('sample-rate').value) || model().capabilities.default_sample_rate);
        } catch {
          await playback.stop('浏览器未能启用实时试听，生成后可播放完整音频');
        }
      } else {
        await playback.stop('等待完整音频生成');
      }
      if (cancelRequested) {
        await playback.stop();
        active = false;
        status('已取消，尚未提交合成任务');
        controls();
        return;
      }
      const url = new URL('/api/synthesis', location.href);
      url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      if (debug) url.searchParams.set('debug', '1');
      socket = new WebSocket(url);
      socket.binaryType = 'arraybuffer';
      status('正在准备合成任务…');
      socket.onopen = () => {
        if (cancelRequested) {
          terminal = true;
          status('已取消，尚未提交合成任务');
          socket.close();
          return;
        }
        socket.send(JSON.stringify({
          type: 'start', config: config(),
          text: mode() === 'incremental' ? '' : $('text').value,
        }));
        log({ type: 'start' });
      };
      socket.onmessage = ({ data }) => {
        if (data instanceof ArrayBuffer) {
          playback.push(data);
          return;
        }
        const message = JSON.parse(data);
        log(message);
        if (message.type === 'ready') {
          ready = true;
          status(mode() === 'incremental' ? '任务已打开，可以发送文本' : '正在生成语音…');
          if (debug) $('effective').textContent = JSON.stringify(message.effective_config, null, 2);
        }
        if (message.metrics) metrics(message.metrics);
        if (message.type === 'input_finished') status('文本输入已结束，正在接收剩余音频…');
        if (message.type === 'completed') {
          terminal = true;
          playback.finish();
          status('生成完成，资源已释放');
          $('player').src = message.audio_url;
          $('player').hidden = false;
          $('download').href = message.audio_url;
          $('download').hidden = false;
          $('result-info').textContent = `${message.metrics.duration_seconds.toFixed(2)} 秒 · WAV · 可重播、下载`;
        }
        if (message.type === 'error' || message.type === 'cancelled') {
          terminal = true;
          void playback.stop();
          if (message.type === 'error') error(message);
          status(message.type === 'cancelled' ? '生成已取消，资源已释放' : '生成失败');
        }
        controls();
      };
      socket.onerror = () => error(new Error('无法连接本机合成服务'));
      socket.onclose = () => {
        if (!terminal) {
          status('连接已断开');
          void playback.stop();
          error(new Error('合成连接中断，请确认服务仍在运行'));
        }
        log({ type: 'connection_closed' });
        active = ready = false;
        socket = null;
        controls();
      };
    } catch (exc) {
      active = false;
      controls();
      await playback.stop();
      throw exc;
    }
  }

  $('model').replaceChildren(...models.map(item => new Option(item.id, item.id)));
  $('model').onchange = defaults;
  root.querySelectorAll('input, select').forEach(el => el.addEventListener('change', controls));
  $('text').oninput = controls;
  $('clear').onclick = () => {
    $('text').value = '';
    controls();
  };
  $('start').onclick = action(start);
  $('send').onclick = action(() => {
    send('text', $('text').value);
    $('text').value = '';
    controls();
  });
  $('finish').onclick = action(() => {
    send('finish_input');
    ended = true;
    status('正在接收剩余音频…');
    controls();
  });
  $('cancel').onclick = action(async () => {
    cancelRequested = true;
    await playback.stop();
    $('player').pause();
    if (socket?.readyState === WebSocket.OPEN) send('cancel');
    ended = true;
    status('正在取消，等待资源释放…');
    controls();
  });
  $('stop').onclick = action(async () => {
    await playback.stop();
    $('player').pause();
  });
  $('player').onplay = () => { void playback.stop('正在播放完整音频'); };
  if (debug) {
    for (const name of ['prepare', 'release']) $(name).onclick = action(async () => {
      active = true;
      controls();
      try {
        const result = await request(`/api/debug/synthesis/${name}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          ...(name === 'prepare' ? { body: JSON.stringify(config()) } : {}),
        });
        log({ type: name, ...result });
        if (result.effective_config) $('effective').textContent = JSON.stringify(result.effective_config, null, 2);
        status(name === 'prepare' ? '配置已就绪；生成时连接云端并验证权限' : '合成模型已释放');
      } finally {
        active = false;
        controls();
      }
    });
    $('export').onclick = () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(events, null, 2)], { type: 'application/json' }));
      const link = document.createElement('a');
      link.href = url;
      link.download = 'speechio-synthesis-events.json';
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
  } else {
    const drawer = visible => {
      $('drawer').hidden = $('backdrop').hidden = !visible;
      (visible ? $('settings-close') : $('settings-open')).focus();
    };
    $('settings-open').onclick = () => drawer(true);
    for (const name of ['settings-close', 'settings-done', 'backdrop']) $(name).onclick = () => drawer(false);
    $('drawer').onkeydown = event => {
      if (event.key === 'Escape') drawer(false);
      if (event.key !== 'Tab') return;
      const elements = [...$('drawer').querySelectorAll('button, input, select, summary')].filter(el => !el.disabled && el.getClientRects().length);
      if (event.shiftKey && document.activeElement === elements[0]) {
        event.preventDefault();
        elements.at(-1).focus();
      }
      if (!event.shiftKey && document.activeElement === elements.at(-1)) {
        event.preventDefault();
        elements[0].focus();
      }
    };
  }
  window.addEventListener('pagehide', () => {
    socket?.close();
    void playback.stop();
  });
  defaults();
  return {
    busy: () => active,
    hide: () => {
      $('player').pause();
      void playback.stop();
    },
  };
}
