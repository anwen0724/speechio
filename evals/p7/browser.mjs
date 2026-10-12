/** 真实 Chrome 公开音源经过采集、网络与模型，验证稳定展示及最终归属。 */
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { writeFile } from "node:fs/promises";
import assert from "node:assert/strict";

const require = createRequire(resolve("web/package.json"));
const { chromium } = require("playwright");
const [url, folder, wav] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    `--use-file-for-fake-audio-capture=${wav}`,
  ],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1280, height: 900 },
  });
  await page.goto(url);
  await page.getByLabel("接入模式").selectOption("manual");
  await page.getByLabel("稳定文本", { exact: true }).check();
  await page.getByRole("button", { name: "连接", exact: true }).click();
  await page.getByText("已连接", { exact: true }).waitFor();
  await page.getByRole("button", { name: "开始采集", exact: true }).click();
  await page.waitForFunction(
    () =>
      Boolean(
        document.querySelector('[aria-label="稳定前缀"]')?.textContent?.trim(),
      ),
    { timeout: 20000 },
  );
  const stable = await page.getByLabel("稳定前缀", { exact: true }).innerText();
  const unstable = await page
    .getByLabel("待定尾部", { exact: true })
    .innerText();
  assert(stable.length > 0);
  await page.screenshot({
    path: resolve(folder, "browser-stable-desktop.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "结束当前输入", exact: true }).click();
  await page.waitForFunction(
    () =>
      Boolean(
        document.querySelector(".finals li")?.textContent?.includes("已完成"),
      ),
    { timeout: 30000 },
  );
  const final = await page.locator(".finals li p").first().innerText();
  assert(final.length > 0);
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
        final,
        source: "public WAV through Chrome microphone simulation",
        human_microphone_accepted: false,
      },
      null,
      2,
    ),
  );
  console.log("real P7 browser stable and final checks passed");
} finally {
  await browser.close();
}
