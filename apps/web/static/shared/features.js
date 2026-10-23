/** 页面功能切换；识别和合成保留独立控件及状态。 */
import { mountSynthesis } from './synthesis.js';

export function mountFeatures(debug, config, asrBusy, closeFinished) {
  const app = document.querySelector('#app');
  const header = app.querySelector('header');
  const asr = document.createElement('section');
  asr.id = 'asr-content';
  for (const child of [...app.children]) {
    if (child !== header) asr.append(child);
  }
  const tts = document.createElement('section');
  tts.id = 'tts-content';
  tts.hidden = true;
  const nav = document.createElement('nav');
  nav.className = 'feature-nav';
  nav.setAttribute('aria-label', '语音功能');
  nav.innerHTML = `<div class="tabs">
    <button data-feature="asr" class="active" aria-pressed="true">语音转文字</button>
    <button data-feature="tts" aria-pressed="false">文字转语音</button>
    </div><span class="help" role="status"></span>`;
  app.append(nav, asr, tts);
  const synthesis = mountSynthesis(tts, debug, config.synthesis_models || []);
  nav.querySelectorAll('button').forEach(button => {
    button.onclick = async () => {
      if (button.classList.contains('active')) return;
      if (asrBusy() || synthesis.busy()) {
        nav.querySelector('span').textContent = '请先结束或取消当前任务，等待资源释放后再切换。';
        return;
      }
      nav.querySelector('span').textContent = '';
      await closeFinished();
      synthesis.hide();
      asr.hidden = button.dataset.feature !== 'asr';
      tts.hidden = !asr.hidden;
      nav.querySelectorAll('button').forEach(item => {
        item.classList.toggle('active', item === button);
        item.setAttribute('aria-pressed', item === button);
      });
    };
  });
  if (!config.models.length) nav.querySelector('[data-feature="tts"]').click();
}
