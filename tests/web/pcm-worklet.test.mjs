/** 浏览器音频线程边界：停止确认必须晚于最后一个不满块的音频消息。 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

test('stop flushes the final partial PCM block before acknowledgement', () => {
  const delivered = [];
  let Processor;
  class AudioWorkletProcessor {
    constructor() {
      this.port = { postMessage: message => delivered.push(message) };
    }
  }
  const scope = vm.createContext({
    AudioWorkletProcessor,
    registerProcessor: (_name, implementation) => { Processor = implementation; },
  });
  vm.runInContext(readFileSync(new URL('../../apps/web/static/shared/pcm-worklet.js', import.meta.url), 'utf8'), scope);
  const processor = new Processor();
  processor.process([[new Float32Array(1024).fill(.5)]]);
  processor.process([[new Float32Array(13).fill(-.5)]]);
  processor.port.onmessage({ data: 'stop' });
  assert.deepEqual(delivered.map(item => item.type), ['audio', 'audio', 'stopped']);
  assert.equal(delivered[0].bytes.byteLength, 2048);
  assert.equal(delivered[1].bytes.byteLength, 26);
  assert.equal(new DataView(delivered[1].bytes).getInt16(24, true), -16384);
});
