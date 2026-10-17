/** 真实 Chrome 公开音源经过采集、网络与模型，验证稳定展示及最终归属。 */
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { writeFile } from "node:fs/promises";
import assert from "node:assert/strict";

const require = createRequire(resolve("apps/web/package.json"));
const { chromium } = require("playwright");
const [url, folder, wav] = process.argv.slice(2);
const dual = process.argv[5] === "dual";
const candidates = process.argv[5] === "candidates";
const wake = process.argv[5] === "wake";
const browser = await chromium.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}`,
  ],
});
let page;
try {
  page = await browser.newPage({
    viewport: { width: 1280, height: 900 },
  });
  page.on('pageerror', (error) => console.error('页面错误：', error.message));
  page.on('websocket', (socket) => socket.on('framereceived', ({payload}) => {
    if (typeof payload !== 'string') return;
    const message = JSON.parse(payload);
    if (message.type === 'error') console.error('服务错误：', message.code, message.message);
  }));
  await page.goto(url);
  await page.getByLabel("接入模式").selectOption(wake ? "auto" : "manual");
  if (wake) await page.getByLabel('等待唤醒词', {exact: true}).check();
  await page.getByLabel("稳定文本", { exact: true }).check();
  if (dual) await page.getByLabel("双实时合并", { exact: true }).check();
  if (candidates) {
    await page.getByLabel("完成策略", {exact: true}).selectOption("candidate");
    await page.getByLabel("保留最近整轮供修正", {exact: true}).check();
  }
  await page.getByRole("button", { name: "连接", exact: true }).click();
  await page.getByText("已连接", { exact: true }).waitFor();
  await page.getByRole("button", { name: "开始采集", exact: true }).click();
  if (wake) {
    await page.waitForFunction(() => document.querySelector('meter')?.value > 0);
    await page.getByLabel('门控状态', {exact: true}).getByText('正在识别正文', {exact: false}).waitFor();
  }
  await page.waitForFunction(
    (needsBody) => {
      const text = document.querySelector('[aria-label="稳定前缀"]')?.textContent?.trim();
      return needsBody ? Boolean(text && /call/i.test(text)) : Boolean(text);
    }, wake, { timeout: 30000 },
  );
  const stable = await page.getByLabel("稳定前缀", { exact: true }).innerText();
  const unstable = await page
    .getByLabel("待定尾部", { exact: true })
    .innerText();
  let sources;
  if (dual) {
    await page.waitForFunction(() => {
      const fast = document.querySelector('[aria-label="极速实时原文"]')?.textContent;
      return Boolean(fast && !fast.includes("等待"));
    }, { timeout: 20000 });
    sources = {
      accurate: await page.getByLabel("准确实时正文", { exact: true }).innerText(),
      ultrafast: await page.getByLabel("极速实时原文", { exact: true }).innerText(),
      merged: await page.getByLabel("合并实时文本", { exact: true }).innerText(),
      status: await page.getByLabel("合并状态", { exact: true }).innerText(),
    };
    assert(sources.accurate.length > 0 && sources.ultrafast.length > 0);
  }
  assert(stable.length > 0);
  let candidate;
  if (candidates) {
    await page.getByRole('button', {name: '整轮候选', exact: true}).click();
    await page.getByLabel('候选正文', {exact: true}).getByText('发布时', {exact: false}).waitFor();
    candidate = await page.getByLabel('候选正文', {exact: true}).innerText();
    await page.getByRole('button', {name: '同轮继续', exact: true}).click();
  }
  await page.screenshot({
    path: resolve(folder, "browser-stable-desktop.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "结束当前输入", exact: true }).click();
  await page.waitForFunction(
    (needsBody) => Array.from(document.querySelectorAll('.finals li')).some((row) =>
      row.textContent.includes('已完成') && (!needsBody || /call/i.test(row.textContent))),
    wake, { timeout: 30000 },
  );
  const final = await page.locator('.finals li').last().locator('p').first().innerText();
  assert(final.length > 0);
  if (candidates) {
    await page.getByRole('button', {name: '修正最近整轮', exact: true}).click();
    await page.getByLabel('候选正文', {exact: true}).getByText('correction', {exact: false}).waitFor();
    assert.equal(await page.locator('.finals li').count(), 1);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  assert(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  );
  await page.screenshot({
    path: resolve(folder, "browser-final-mobile.png"),
    fullPage: true,
  });
  if (wake) {
    assert(await page.getByLabel('会话时间线', {exact: true}).locator('li').count() > 0);
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.getByRole('button', {name: '复制文本', exact: true}).click();
    const copied = await page.evaluate(() => navigator.clipboard.readText());
    assert(copied.includes(final));
    await page.getByRole('button', {name: '清除展示', exact: true}).click();
    assert.equal(await page.locator('.finals li').count(), 0);
  }
  await page.getByRole("button", { name: "重置会话", exact: true }).click();
  assert.equal(await page.getByLabel("稳定前缀").count(), 0);
  await page.getByRole("button", { name: "断开", exact: true }).click();
  await page.getByText("已关闭", { exact: true }).waitFor();
  await writeFile(
    resolve(folder, "browser.json"),
    JSON.stringify(
      {
        status: "passed",
        stable,
        unstable,
        dual,
        candidates,
        wake,
        candidate,
        sources,
        final,
        source: "public WAV through Chrome microphone simulation",
        human_microphone_accepted: false,
      },
      null,
      2,
    ),
  );
  console.log("real P7 browser stable and final checks passed");
} catch (error) {
  if (page) {
    await page.screenshot({path: resolve(folder, 'browser-failure.png'), fullPage: true});
    await writeFile(resolve(folder, 'browser-failure.txt'), await page.locator('body').innerText());
  }
  throw error;
} finally {
  await browser.close();
}
