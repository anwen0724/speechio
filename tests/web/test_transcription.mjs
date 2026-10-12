/** 浏览器通过可控的外部 WebSocket 验证迟到最终不会清掉另一单元的预览。 */
import { createRequire } from "node:module";
import { resolve } from "node:path";
import assert from "node:assert/strict";

const require = createRequire(resolve("web/package.json"));
const { chromium } = require("playwright");
const browser = await chromium.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
  ],
});
async function run() {
  const page = await browser.newPage();
  const commands = [];
  await page.routeWebSocket("**/api/v1/ws", (socket) => {
    let sequence = 0;
    const send = (message) =>
      socket.send(
        JSON.stringify({
          protocol_version: 1,
          session_id: "browser-test",
          generation: 0,
          wire_sequence: ++sequence,
          ...message,
        }),
      );
    const unit = (unit_id) => ({
      session_id: "browser-test",
      generation: 0,
      unit_id,
    });
    socket.onMessage((payload) => {
      if (typeof payload !== "string") return;
      const command = JSON.parse(payload);
      commands.push(command.type);
      if (process.argv[3] === "auth-error" && command.type === "open") {
        send({
          type: "error",
          code: "UNAUTHORIZED",
          message: "需要有效 Bearer Token",
        });
        return;
      }
      if (command.type === "open")
        send({ type: "ready", request_id: command.request_id });
      if (command.type === "open" && process.argv[3] === "stable")
        assert.equal(command.stable_text, true);
      if (command.type === "end_input")
        send({
          type: "ack",
          command: "end_input",
          request_id: command.request_id,
        });
      if (process.argv[3] === "auto-eof" && command.type === "finish") {
        send({
          type: "error",
          request_id: command.request_id,
          code: "NO_ACTIVE_UNIT",
          message: "没有活动段",
        });
      }
      if (command.type === "start") {
        send({
          type: "ack",
          command: "start",
          request_id: command.request_id,
          unit: unit(2),
        });
        if (process.argv[3] === "realtime-failure") {
          send({
            type: "event",
            event: {
              kind: "realtime_failed",
              sequence: 1,
              unit: unit(2),
              result: null,
              error: { code: "INFERENCE_FAILED", message: "实时观察失败" },
            },
          });
          return;
        }
        send({
          type: "event",
          event: {
            kind: "realtime",
            sequence: 1,
            unit: unit(2),
            result: { text: "正在说第二句" },
            error: null,
            ...(process.argv[3] === "stable"
              ? {
                  stabilization: {
                    accepted: true,
                    stable_text: "正在说",
                    unstable_text: "第二句",
                    display_text: "正在说第二句",
                    stable_prefix_conflict: false,
                  },
                }
              : {}),
          },
        });
        send({
          type: "event",
          event: {
            kind: "final",
            sequence: 2,
            unit: unit(1),
            result: { text: "上一句话" },
            error: null,
            ...(process.argv[3] === "stable"
              ? {
                  finalization: {
                    agrees_with_stable_prefix: false,
                    mismatch_reason: "stable-prefix-mismatch",
                  },
                }
              : {}),
          },
        });
      }
    });
  });
  await page.goto(process.argv[2]);
  await page
    .getByLabel("接入模式")
    .selectOption(process.argv[3] === "auto-eof" ? "auto" : "manual");
  if (process.argv[3] === "stable")
    await page.getByLabel("稳定文本", { exact: true }).check();
  await page.getByRole("button", { name: "连接", exact: true }).click();
  if (process.argv[3] === "auth-error") {
    await page.getByRole("alert").waitFor();
    assert(
      (await page.getByRole("alert").innerText()).includes("UNAUTHORIZED"),
    );
    console.log("authentication failure preserves original cause: passed");
    return;
  }
  await page.getByText("已连接", { exact: true }).waitFor();
  await page.getByRole("button", { name: "开始采集", exact: true }).click();
  if (process.argv[3] === "auto-eof") {
    await page.getByText("采集中", { exact: true }).waitFor();
    await page
      .getByRole("button", { name: "结束当前输入", exact: true })
      .click();
    await page
      .locator(".status")
      .filter({ hasText: /已连接|错误/ })
      .waitFor();
    assert(
      commands.includes("end_input"),
      "自动源停止必须使用 EOF 刷新核心短尾",
    );
    assert.equal(await page.locator(".status").innerText(), "已连接");
    console.log("automatic stop seals source EOF: passed");
  } else if (process.argv[3] === "realtime-failure") {
    await page.getByRole("alert").waitFor();
    assert.equal(await page.locator(".status").innerText(), "采集中");
    assert(
      await page
        .getByRole("button", { name: "结束当前输入", exact: true })
        .isEnabled(),
    );
    console.log("realtime failure preserves input and final path: passed");
  } else {
    await page.getByText("上一句话", { exact: true }).waitFor();
    assert.equal(
      await page.locator(".preview-panel p").innerText(),
      "正在说第二句",
    );
    if (process.argv[3] === "stable") {
      assert.equal(
        await page.getByLabel("稳定前缀", { exact: true }).innerText(),
        "正在说",
      );
      assert.equal(
        await page.getByLabel("待定尾部", { exact: true }).innerText(),
        "第二句",
      );
      await page
        .getByText("最终正文与实时稳定前缀不同，以上保留最终识别结果。")
        .waitFor();
    }
    console.log("late final preserves current preview: passed");
  }
}
try {
  await run();
} finally {
  await browser.close();
}
