/** 用公开录音作为浏览器虚拟设备，仍走 AudioWorklet、网络与真实模型。 */
import { createRequire } from 'node:module';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';

const require = createRequire(resolve('web/package.json'));
const { chromium } = require('playwright');
const [url, audio, folder] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true,
  args: [
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
    `--use-file-for-fake-audio-capture=${audio}`,
  ],
});
const errors = [];
const packets = [];
const messages = [];
try {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    permissions: ['microphone', 'clipboard-write'],
  });
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('websocket', (socket) => {
    socket.on('framesent', ({ payload }) => {
      if (typeof payload !== 'string') {
        const length = payload.readUInt32LE(0);
        packets.push(JSON.parse(payload.subarray(4, 4 + length).toString()));
      }
    });
    socket.on('framereceived', ({ payload }) => messages.push(JSON.parse(String(payload))));
  });
  await page.goto(url);
  await page.getByLabel('接入模式').selectOption('manual');
  await page.getByRole('button', { name: '连接', exact: true }).click();
  await page.getByText('已连接', { exact: true }).waitFor();
  await page.getByRole('button', { name: '开始采集', exact: true }).click();
  await page.getByText('采集中', { exact: true }).waitFor();
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    const sourceFrames = packets.reduce((sum, item) => sum + item.frame_count, 0);
    const rate = packets[0]?.format.sample_rate;
    if (rate && sourceFrames >= rate * 8) break;
    await page.waitForTimeout(50);
  }
  assert(packets.length > 0, '浏览器没有递交音频');
  await page.getByRole('button', { name: '结束当前输入', exact: true }).click();
  await page.getByText('已连接', { exact: true }).waitFor();
  await page.locator('.finals li').first().waitFor({ timeout: 30000 });
  const text = await page.locator('.finals li p').first().innerText();
  assert(text && text !== '（无识别正文）', '真实模型没有产生正文');
  const firstGeneration = messages.find((item) => item.type === 'ready').generation;
  await page.screenshot({ path: resolve(folder, 'browser-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: resolve(folder, 'browser-mobile.png'), fullPage: true });
  await page.getByRole('button', { name: '重置会话', exact: true }).click();
  await page.getByText('已连接', { exact: true }).waitFor();
  await page.getByRole('button', { name: '开始采集', exact: true }).click();
  await page.getByText('采集中', { exact: true }).waitFor();
  await page.getByRole('button', { name: '取消当前单元', exact: true }).click();
  await page.getByText('已连接', { exact: true }).waitFor();
  await page.getByRole('button', { name: '断开', exact: true }).click();
  await page.getByText('已关闭', { exact: true }).waitFor();
  assert(
    messages.some((item) => item.command === 'reset' && item.generation === firstGeneration + 1),
  );
  const finals = messages.filter((item) => item.event?.kind === 'final');
  assert.equal(finals.length, 1, '取消单元产生了最终');
  assert(
    packets.every((item) => item.format.sample_type === 'float32' && item.format.channels === 1),
  );
  assert(errors.length === 0, errors.join('\n'));
  const sourceFrames = packets
    .filter((item) => item.generation === firstGeneration)
    .reduce((sum, item) => sum + item.frame_count, 0);
  const finalSamples = finals[0].event.audio_samples;
  assert(
    Math.abs(finalSamples - (sourceFrames * 16000) / packets[0].format.sample_rate) <= 1,
    'Worklet 短尾丢失',
  );
  // 同一页面重新取得设备与会话，不复用旧 socket 的归属。
  await page.getByRole('button', { name: '连接', exact: true }).click();
  await page.getByText('已连接', { exact: true }).waitFor();
  await page.getByRole('button', { name: '断开', exact: true }).click();
  await page.getByText('已关闭', { exact: true }).waitFor();
  await writeFile(
    resolve(folder, 'browser.json'),
    JSON.stringify(
      {
        text,
        sourceFrames,
        finalSamples,
        packetCount: packets.length,
        shortTail: packets.some((item) => item.frame_count < 1024),
        desktop: true,
        mobile: true,
        resetCancelReopen: true,
        errors,
        messages,
      },
      null,
      2,
    ),
  );
  await context.close();
} finally {
  await browser.close();
}
