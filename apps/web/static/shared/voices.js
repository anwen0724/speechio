/** 音色按可用名称统一展示，来源与文件记录由应用处理。 */
import { request } from './client.js';
import { mountDirectoryPicker } from './voice-directory.js';

export function mountVoices(root, hooks) {
  root.innerHTML = `<nav class="voice-tabs" aria-label="语音功能">
      <button id="voice-synthesize" aria-pressed="true">生成语音</button>
      <button id="voice-clone-tab" aria-pressed="false">克隆音色</button>
    </nav>
    <section class="voice-panel">
      <div class="voice-heading"><h2 id="voice-title">选择声音</h2><button class="quiet" id="voice-back" hidden>返回生成语音</button></div>
      <div class="voice-selection"><div id="voice-model-slot"></div>
        <label class="field" id="voice-choice-field"><span class="field-title">音色</span><select id="voice-choice"></select></label>
      </div>
      <div id="voice-device-slot"></div>
      <div id="voice-selection-tools">
        <p class="help" id="voice-empty" hidden>这个模型还没有可用音色，请先克隆或导入音色。</p>
        <div class="voice-links"><button id="voice-direct">用参考录音生成</button><button id="voice-refresh">刷新音色</button></div>
        <details class="voice-file-tools"><summary>音色文件与其他音色</summary>
          <div id="voice-selected-file" hidden><p class="help voice-path" id="voice-file-path"></p>
            <div class="tts-actions"><a class="quiet" id="voice-download">下载音色文件</a><button class="quiet" id="voice-remove">删除本机音色</button></div></div>
          <div id="voice-custom-field"><label class="field"><span class="field-title">其他音色 ID</span><input id="voice-custom" placeholder="模型支持的其他音色 ID"></label><button class="quiet" id="voice-custom-use">使用此音色</button></div>
          <div id="voice-import-fields">
            <label class="field"><span class="field-title">导入音色文件</span><input id="voice-import-file" type="file" accept=".voice"></label>
            <label class="field"><span class="field-title">音色名称</span><input id="voice-import-name" maxlength="100" placeholder="留空使用文件名"></label>
            <button class="quiet" id="voice-import">导入音色</button>
          </div>
        </details>
      </div>
      <div id="voice-reference" hidden>
        <div class="voice-heading"><h3>参考录音</h3><button class="quiet" id="voice-choose-existing">返回选择音色</button></div>
        <label class="voice-upload field"><span class="field-title">选择一段声音清晰的录音</span><input id="voice-audio" type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.opus,.aac,.wma,.aiff,.aif"></label>
        <audio id="voice-preview" controls hidden></audio><p class="help" id="voice-hint"></p>
        <label class="field" id="voice-reference-text-field"><span class="field-title">录音中的文字（可选）</span><textarea id="voice-reference-text" rows="2" placeholder="填写与参考录音一致的文字"></textarea></label>
        <p class="help" id="voice-direct-note">这次生成使用录音中的声音，完整生成后播放，不保存音色。</p>
      </div>
      <div id="voice-create" hidden>
        <label class="field"><span class="field-title">音色名称</span><input id="voice-name" maxlength="100" placeholder="例如：御姐音1"></label>
        <div class="voice-save-location"><label class="field"><span class="field-title">保存目录</span><input id="voice-save-dir" placeholder="输入本机目录的完整路径"></label><button class="quiet" id="voice-browse">选择目录</button></div>
        <p class="help">保存到运行 Web 服务的电脑。新目录会自动创建，下次打开仍可选择这个音色。</p>
        <div class="tts-actions"><button class="primary" id="voice-clone">克隆并保存</button></div>
      </div>
      <div class="voice-success" id="voice-success" role="status" hidden>
        <strong id="voice-success-title"></strong><p class="help voice-path" id="voice-success-path"></p><button class="quiet" id="voice-use">使用这个音色</button>
      </div>
    </section><div id="voice-directory"></div>`;
  const $ = name => root.querySelector(`#voice-${name}`);
  $('model-slot').append(hooks.modelField);
  $('device-slot').append(hooks.deviceField);
  let cloning = false, direct = false, busy = false, cloneRunning = false;
  let records = [], previewUrl = null, latestVoice = null, loadedDirectory = false;
  const source = () => cloning || direct ? 'reference' : $('choice').value.startsWith('saved:') ? 'saved' : 'preset';
  const selected = () => records.find(record => `saved:${record.id}` === $('choice').value);
  const picker = mountDirectoryPicker($('directory'), path => {
    $('save-dir').value = path;
    hooks.changed();
  });

  /** 返回只切换视图，不取消正在执行的克隆任务。 */
  function update() {
    const model = hooks.model(), cap = Boolean(model?.capabilities.voice_cloning);
    $('device-slot').hidden = cloning || model?.source !== 'local';
    $('title').textContent = cloning ? '创建你的音色' : direct ? '使用参考录音' : '选择声音';
    hooks.modelField.querySelector('.field-title').textContent = cloning ? '克隆模型' : '模型';
    hooks.modelField.querySelectorAll('option').forEach(option => {
      option.hidden = option.disabled = cloning && !hooks.models.find(item => item.id === option.value)?.capabilities.voice_cloning;
    });
    $('clone-tab').hidden = !hooks.models.some(item => item.capabilities.voice_cloning);
    $('back').hidden = !cloning;
    $('choice-field').hidden = $('selection-tools').hidden = cloning || direct;
    $('reference').hidden = !cloning && !direct;
    $('create').hidden = !cloning;
    $('choose-existing').hidden = $('direct-note').hidden = cloning;
    $('reference-text-field').hidden = !model?.voice_inputs.reference_text;
    $('hint').textContent = model?.voice_inputs.reference_hint || '';
    $('direct').hidden = $('import-fields').hidden = !cap;
    $('custom-field').hidden = !model?.voice_inputs.preset;
    $('empty').hidden = Boolean($('choice').value);
    $('synthesize').setAttribute('aria-pressed', String(!cloning));
    $('clone-tab').setAttribute('aria-pressed', String(cloning));
    root.querySelectorAll('.voice-panel input, .voice-panel select, .voice-panel textarea, .voice-panel button').forEach(element => { element.disabled = busy; });
    $('synthesize').disabled = $('back').disabled = false;
    $('clone-tab').disabled = busy && !cloneRunning;
    $('clone').disabled = busy || !cap || !$('audio').files.length || !$('name').value.trim() || !$('save-dir').value.trim();
    $('clone').textContent = cloneRunning ? '正在克隆…' : '克隆并保存';
    $('import').disabled = busy || !$('import-file').files.length;
    $('custom-use').disabled = busy || !$('custom').value.trim();
    $('success').hidden = !latestVoice;
    if (latestVoice) {
      $('success-title').textContent = `“${latestVoice.name}”已保存`;
      $('success-path').textContent = latestVoice.file_path;
      $('use').textContent = `使用“${latestVoice.name}”`;
    }
    const record = selected();
    $('selected-file').hidden = !record;
    if (record) {
      $('download').href = `/api/voices/${record.id}/download`;
      $('file-path').textContent = record.file_path;
    }
  }

  function changed() {
    update();
    hooks.changed();
  }

  /** 一张列表展示可用音色，来源区别仅用于构造服务端请求。 */
  function choices(preferred) {
    const model = hooks.model(), current = preferred || $('choice').value;
    const presets = model?.voice_inputs.preset
      ? [...new Set([model.defaults.voice, ...model.voices, hooks.presetVoice()].filter(Boolean))] : [];
    const options = [
      ...presets.map(name => new Option(name, `preset:${name}`)),
      ...records.map(record => new Option(record.name, `saved:${record.id}`)),
    ];
    $('choice').replaceChildren(...(options.length ? options : [new Option('暂无可用音色', '')]));
    if (options.some(option => option.value === current)) $('choice').value = current;
    if (source() === 'preset') hooks.setPresetVoice($('choice').value.slice('preset:'.length));
  }

  async function refresh(preferred) {
    const model = hooks.model();
    if (!model) return;
    const value = await request(`/api/voices?model=${encodeURIComponent(model.id)}`);
    if (model.id !== hooks.model()?.id) return;
    records = value.voices;
    if (!loadedDirectory) {
      $('save-dir').value = value.save_dir;
      loadedDirectory = true;
    }
    choices(preferred);
    changed();
  }

  async function upload(file) {
    if (!file) throw new Error('请先选择文件');
    return (await request('/api/uploads', {
      method: 'POST', body: file,
      headers: { 'X-Audio-Extension': file.name.match(/\.[a-z0-9]+$/i)?.[0] || '.audio' },
    })).token;
  }
  function referenceText() {
    return hooks.model()?.voice_inputs.reference_text ? $('reference-text').value.trim() || null : null;
  }
  async function task(fn) {
    hooks.busy(true);
    try {
      await fn();
    } catch (error) {
      hooks.error(error);
      hooks.status('音色操作失败，请查看错误提示。');
      hooks.log({ type: 'voice_error', message: error.message });
    } finally {
      cloneRunning = false;
      hooks.busy(false);
    }
  }
  $('choice').onchange = () => {
    if (source() === 'preset') hooks.setPresetVoice($('choice').value.slice('preset:'.length));
    changed();
  };
  $('synthesize').onclick = $('back').onclick = () => {
    cloning = false;
    if (!busy) hooks.status('选择音色并输入文字。');
    changed();
  };
  $('clone-tab').onclick = () => {
    cloning = true;
    direct = false;
    if (!hooks.model()?.capabilities.voice_cloning) {
      hooks.chooseModel(hooks.models.find(item => item.capabilities.voice_cloning).id);
    }
    if (!busy) hooks.status('选择克隆模型，上传录音并保存音色。');
    changed();
  };
  $('direct').onclick = () => {
    direct = true;
    hooks.status('上传参考录音，再输入需要朗读的文字。');
    changed();
  };
  $('choose-existing').onclick = () => {
    direct = false;
    changed();
  };
  $('use').onclick = () => {
    cloning = direct = false;
    hooks.chooseModel(latestVoice.model_key);
    changed();
    void refresh(`saved:${latestVoice.id}`).catch(hooks.error);
  };
  $('refresh').onclick = () => { void refresh().catch(hooks.error); };
  $('custom-use').onclick = () => {
    const name = $('custom').value.trim();
    hooks.setPresetVoice(name);
    choices(`preset:${name}`);
    changed();
  };
  $('browse').onclick = () => picker.open();
  $('audio').onchange = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = $('audio').files[0] ? URL.createObjectURL($('audio').files[0]) : null;
    $('preview').pause();
    $('preview').hidden = !previewUrl;
    if (previewUrl) $('preview').src = previewUrl;
    else $('preview').removeAttribute('src');
    changed();
  };
  root.querySelectorAll('.voice-panel input, .voice-panel textarea').forEach(element => element.addEventListener('input', changed));
  $('clone').onclick = () => {
    // 上传前固定参数，避免返回页面影响正在执行的请求。
    const file = $('audio').files[0];
    const payload = {
      model: hooks.model().id, name: $('name').value.trim(),
      reference_text: referenceText(), save_dir: $('save-dir').value.trim(), ...hooks.cloneOptions(),
    };
    cloneRunning = true;
    void task(async () => {
      hooks.status('正在上传参考录音并克隆，可以返回生成语音页面查看。');
      const token = await upload(file);
      payload.reference_audio_token = token;
      hooks.log({ type: 'clone_requested', ...payload });
      try {
        const result = await request(`/api/voices/clone${hooks.debug ? '?debug=true' : ''}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
        });
        latestVoice = result.voice;
        await refresh(`saved:${latestVoice.id}`);
        hooks.log({ type: 'voice_saved', ...result });
        hooks.result(result);
        hooks.status(`音色“${latestVoice.name}”已保存，可以直接使用。`);
      } finally {
        await request(`/api/uploads/${token}`, { method: 'DELETE' }).catch(() => {});
      }
    });
  };
  $('import').onclick = () => task(async () => {
    const file = $('import-file').files[0];
    const payload = { model: hooks.model().id, name: $('import-name').value.trim() || file.name.replace(/\.voice$/i, '') };
    const token = await upload(file);
    try {
      const result = await request('/api/voices/import', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, file_token: token }),
      });
      latestVoice = result.voice;
      await refresh(`saved:${result.voice.id}`);
      hooks.status(`已导入音色“${result.voice.name}”`);
      hooks.log({ type: 'voice_imported', ...result });
    } finally {
      await request(`/api/uploads/${token}`, { method: 'DELETE' }).catch(() => {});
    }
  });
  $('remove').onclick = () => {
    const record = selected();
    if (!window.confirm(`删除本机音色“${record.name}”及其文件？云端音色不会被删除。`)) return;
    void task(async () => {
      await request(`/api/voices/${record.id}`, { method: 'DELETE' });
      if (latestVoice?.id === record.id) latestVoice = null;
      await refresh();
      hooks.status(`已删除本机音色“${record.name}”`);
    });
  };
  window.addEventListener('pagehide', () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });
  return {
    source, cloning: () => cloning, savedId: () => selected()?.id || '',
    saveDir: () => $('save-dir').value.trim(), hasVoice: () => Boolean($('choice').value),
    referenceText, hasReference: () => Boolean($('audio').files.length),
    uploadReference: () => upload($('audio').files[0]),
    setBusy(value) {
      busy = value;
      update();
    },
    stopPreview() { $('preview').pause(); },
    modelChanged() {
      direct = false;
      records = [];
      $('choice').replaceChildren();
      choices();
      update();
      void refresh().catch(hooks.error);
    },
  };
}
