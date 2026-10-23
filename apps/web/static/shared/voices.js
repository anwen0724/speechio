/** 两个页面共用的音色操作；只提交模型 ID 与上传令牌。 */
import { request } from './client.js';

export function mountVoices(root, hooks) {
  root.innerHTML = `<div class="toolbar voice-tabs" role="group" aria-label="合成与克隆">
      <button class="quiet" id="voice-synthesize" aria-pressed="true">生成语音</button>
      <button class="quiet" id="voice-clone-tab" aria-pressed="false">克隆音色</button>
    </div>
    <section class="voice-panel">
      <p class="help" id="voice-model"></p>
      <label class="field" id="voice-source-field"><span class="field-title">音色来源</span>
        <select id="voice-source"><option value="preset">预设音色</option><option value="saved">已保存音色</option><option value="reference">参考音频</option></select>
      </label>
      <div id="voice-reference" hidden>
        <label class="field"><span class="field-title">参考音频</span><input id="voice-audio" type="file" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.opus,.aac,.wma,.aiff,.aif"></label>
        <audio id="voice-preview" controls hidden></audio>
        <p class="help" id="voice-hint"></p>
        <label class="field" id="voice-reference-text-field"><span class="field-title">参考录音中的文字（可选）</span><textarea id="voice-reference-text" rows="2" placeholder="与参考录音内容一致；留空也可克隆"></textarea></label>
        <p class="help" id="voice-direct-note">先克隆再完整生成，不自动保存音色。需要反复使用时，请先选择“克隆音色”。</p>
      </div>
      <div id="voice-create" hidden>
        <label class="field"><span class="field-title">音色名称</span><input id="voice-name" maxlength="100" placeholder="例如：日常朗读"></label>
        <button class="primary" id="voice-clone">克隆并保存音色</button>
        <p class="help">克隆期间请等待任务完成；关闭页面不保证终止模型调用。</p>
      </div>
      <div id="voice-library" hidden>
        <label class="field"><span class="field-title">此模型的已保存音色</span><select id="voice-saved"></select></label>
        <p class="help" id="voice-info"></p>
        <div class="tts-actions"><button class="quiet" id="voice-use">使用此音色</button><button class="quiet" id="voice-refresh">刷新列表</button>
          <a class="quiet" id="voice-download" hidden>下载音色文件</a><button class="quiet" id="voice-remove">移除本地记录</button></div>
        <details><summary>导入已有音色</summary><p class="help">选择 SpeechIO 保存的 .voice 文件，导入到当前模型。模型仍会检查文件是否兼容。</p>
          <label class="field"><span class="field-title">音色文件</span><input id="voice-import-file" type="file" accept=".voice"></label>
          <label class="field"><span class="field-title">导入后的名称</span><input id="voice-import-name" maxlength="100" placeholder="留空使用文件名"></label>
          <button class="quiet" id="voice-import">导入音色</button>
        </details>
        <p class="help">音色保存在本机，重启后仍可用。移除本地记录不会删除云端音色。</p>
      </div>
    </section>`;
  const $ = name => root.querySelector(`#voice-${name}`);
  let cloning = false, busy = false, records = [], previewUrl = null;
  const source = () => cloning ? 'reference' : $('source').value;
  const selected = () => records.find(record => record.id === $('saved').value);

  function update() {
    const model = hooks.model(), cap = model?.capabilities.voice_cloning;
    $('model').textContent = model ? `当前模型：${model.id}` : '请先配置合成模型';
    $('clone-tab').hidden = !cap;
    $('source-field').hidden = cloning;
    $('reference').hidden = !cloning && source() !== 'reference';
    $('create').hidden = !cloning;
    $('direct-note').hidden = cloning;
    $('library').hidden = !cap || (!cloning && source() !== 'saved');
    $('reference-text-field').hidden = !model?.voice_inputs.reference_text;
    $('hint').textContent = model?.voice_inputs.reference_hint || '';
    $('synthesize').setAttribute('aria-pressed', String(!cloning));
    $('clone-tab').setAttribute('aria-pressed', String(cloning));
    root.querySelectorAll('input, select, textarea, button').forEach(element => { element.disabled = busy; });
    $('clone').disabled = busy || !model || !$('audio').files.length || !$('name').value.trim();
    $('import').disabled = busy || !$('import-file').files.length;
    $('use').disabled = $('remove').disabled = busy || !selected();
    $('download').hidden = !selected();
    if (selected()) {
      $('download').href = `/api/voices/${selected().id}/download`;
      $('info').textContent = `${selected().model} · ${new Date(selected().created_at).toLocaleString()}`;
    } else {
      $('info').textContent = records.length
        ? '请选择要使用的音色。'
        : '尚无可用音色。可以先克隆，或导入已有音色文件。';
    }
  }

  async function refresh(preferred) {
    const model = hooks.model();
    if (!model) return;
    const value = await request(`/api/voices?model=${encodeURIComponent(model.id)}`);
    if (model.id !== hooks.model()?.id) return;
    const current = preferred || $('saved').value;
    records = value.voices;
    $('saved').replaceChildren(new Option('请选择已保存音色', ''), ...records.map(record => new Option(record.name, record.id)));
    if (records.some(record => record.id === current)) $('saved').value = current;
    update();
    hooks.changed();
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
      hooks.busy(false);
    }
  }

  const changed = () => {
    update();
    hooks.changed();
  };
  $('source').onchange = changed;
  $('saved').onchange = changed;
  $('synthesize').onclick = () => {
    cloning = false;
    changed();
  };
  $('clone-tab').onclick = () => {
    cloning = true;
    hooks.status('请选择模型、上传参考音频并填写音色名称。');
    changed();
  };
  $('use').onclick = () => {
    cloning = false;
    $('source').value = 'saved';
    changed();
  };
  $('refresh').onclick = () => {
    void refresh().catch(hooks.error);
  };
  $('audio').onchange = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = $('audio').files[0] ? URL.createObjectURL($('audio').files[0]) : null;
    $('preview').pause();
    $('preview').hidden = !previewUrl;
    if (previewUrl) $('preview').src = previewUrl;
    else $('preview').removeAttribute('src');
    changed();
  };
  root.querySelectorAll('input, textarea').forEach(element => element.addEventListener('input', changed));
  $('clone').onclick = () => task(async () => {
    hooks.status('正在上传参考音频并克隆，请等待…');
    const token = await upload($('audio').files[0]);
    const payload = {
      model: hooks.model().id,
      name: $('name').value.trim(),
      reference_audio_token: token,
      reference_text: referenceText(),
      ...hooks.cloneOptions(),
    };
    hooks.log({ type: 'clone_requested', ...payload });
    try {
      const result = await request(`/api/voices/clone${hooks.debug ? '?debug=true' : ''}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      await refresh(result.voice.id);
      hooks.log({ type: 'voice_saved', ...result });
      hooks.result(result);
      hooks.status(`音色“${result.voice.name}”已保存，用时 ${result.elapsed_seconds.toFixed(2)} 秒。点击“使用此音色”即可合成。`);
    } finally {
      await request(`/api/uploads/${token}`, { method: 'DELETE' }).catch(() => {});
    }
  });
  $('import').onclick = () => task(async () => {
    const file = $('import-file').files[0];
    const token = await upload(file);
    try {
      const result = await request('/api/voices/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: hooks.model().id,
          name: $('import-name').value.trim() || file.name.replace(/\.voice$/i, ''),
          file_token: token,
        }),
      });
      await refresh(result.voice.id);
      hooks.status(`已导入音色“${result.voice.name}”`);
      hooks.log({ type: 'voice_imported', ...result });
    } finally {
      await request(`/api/uploads/${token}`, { method: 'DELETE' }).catch(() => {});
    }
  });
  $('remove').onclick = () => task(async () => {
    const record = selected();
    await request(`/api/voices/${record.id}`, { method: 'DELETE' });
    await refresh();
    hooks.status(`已移除“${record.name}”的本地记录`);
  });
  window.addEventListener('pagehide', () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });

  return {
    source,
    cloning: () => cloning,
    savedId: () => $('saved').value,
    referenceText,
    hasReference: () => Boolean($('audio').files.length),
    uploadReference: () => upload($('audio').files[0]),
    setBusy(value) {
      busy = value;
      update();
    },
    stopPreview() {
      $('preview').pause();
    },
    modelChanged() {
      const model = hooks.model(), cap = model?.capabilities.voice_cloning;
      cloning = cloning && Boolean(cap);
      [...$('source').options].forEach(option => {
        option.hidden = option.disabled = option.value === 'preset' ? !model?.voice_inputs.preset : !cap;
      });
      $('source').value = model?.voice_inputs.preset ? 'preset' : 'reference';
      records = [];
      $('saved').replaceChildren();
      update();
      void refresh().catch(hooks.error);
    },
  };
}
