/** 本机目录选择器；浏览结果来自运行 Web 服务的电脑。 */
import { request } from './client.js';

export function mountDirectoryPicker(root, onChoose) {
  root.innerHTML = `<dialog class="voice-directory-dialog" aria-labelledby="voice-directory-title">
    <div class="section-head"><h2 id="voice-directory-title">选择音色保存目录</h2><button class="quiet" data-close aria-label="关闭目录选择">关闭</button></div>
    <div class="directory-location">
      <label class="field"><span class="field-title">磁盘</span><select data-roots></select></label>
      <label class="field"><span class="field-title">当前目录</span><input data-path></label>
      <button class="quiet" data-open>打开</button>
    </div>
    <button class="quiet" data-parent>上一级</button>
    <p class="help" data-error role="alert" hidden></p>
    <div class="directory-list" data-list></div>
    <div class="tts-actions"><button class="primary" data-choose>选择此目录</button><button class="quiet" data-cancel>取消</button></div>
  </dialog>`;
  const dialog = root.querySelector('dialog');
  const $ = name => root.querySelector(`[data-${name}]`);
  let current;
  async function browse(path) {
    $('error').hidden = true;
    try {
      const listing = await request('/api/voice-directories' + (path ? `?path=${encodeURIComponent(path)}` : ''));
      current = listing;
      $('path').value = listing.path;
      $('roots').replaceChildren(...listing.roots.map(value => new Option(value, value)));
      $('roots').value = listing.roots.find(value => listing.path.startsWith(value)) || '';
      $('parent').disabled = !listing.parent;
      $('choose').disabled = false;
      $('list').replaceChildren(...listing.directories.map(item => {
        const button = document.createElement('button');
        button.className = 'directory-entry';
        button.textContent = item.name;
        button.onclick = () => { void browse(item.path); };
        return button;
      }));
      if (!listing.directories.length) $('list').textContent = '此目录下没有子文件夹';
    } catch (error) {
      $('error').textContent = error.message;
      $('error').hidden = false;
      $('choose').disabled = true;
    }
  }
  $('open').onclick = () => { void browse($('path').value.trim()); };
  $('path').onkeydown = event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      void browse($('path').value.trim());
    }
  };
  $('path').oninput = () => { $('choose').disabled = true; };
  $('roots').onchange = () => { void browse($('roots').value); };
  $('parent').onclick = () => { void browse(current.parent); };
  $('choose').onclick = () => {
    onChoose(current.path);
    dialog.close();
  };
  for (const name of ['close', 'cancel']) $(name).onclick = () => dialog.close();
  return {
    open() {
      dialog.showModal();
      $('choose').disabled = true;
      void browse();
    },
  };
}
