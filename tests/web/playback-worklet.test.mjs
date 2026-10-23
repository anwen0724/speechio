/** 播放边界：保留尾部采样，完成通知晚于最后一块音频。 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

test('PCM playback preserves ordered samples and tail at browser sample rates', () => {
  for (const rate of [24000, 48000]) {
    const messages = [];
    let Processor;
    class AudioWorkletProcessor {
      constructor() { this.port = { postMessage: message => messages.push(message) }; }
    }
    const scope = vm.createContext({
      AudioWorkletProcessor, sampleRate: rate,
      registerProcessor: (_name, implementation) => { Processor = implementation; },
    });
    vm.runInContext(readFileSync(new URL('../../apps/web/static/shared/playback-worklet.js', import.meta.url), 'utf8'), scope);
    const processor = new Processor({ processorOptions: { sampleRate: 24000 } });
    const first = new Int16Array(128).fill(16384);
    const tail = new Int16Array(13).fill(-16384);
    processor.port.onmessage({ data: { audio: first.buffer } });
    processor.port.onmessage({ data: { audio: tail.buffer } });
    processor.port.onmessage({ data: { type: 'finish' } });
    const samples = [];
    let keep;
    do {
      const output = new Float32Array(128);
      keep = processor.process([], [[output]]);
      samples.push(...output);
    } while (keep);
    assert.equal(samples[0], .5);
    const total = 141 * rate / 24000;
    assert.equal(samples[total - 1], -.5);
    assert.ok(samples.slice(total).every(value => value === 0));
    assert.equal(messages.at(-1).type, 'ended');
    assert.equal(messages.filter(item => item.type === 'consumed').reduce((n, item) => n + item.samples, 0), 141);
  }
});

test('playback rejects audio beyond its bounded queue', () => {
  const messages = [];
  let Processor;
  class AudioWorkletProcessor {
    constructor() { this.port = { postMessage: message => messages.push(message) }; }
  }
  const scope = vm.createContext({ AudioWorkletProcessor, sampleRate: 24000,
    registerProcessor: (_name, implementation) => { Processor = implementation; } });
  vm.runInContext(readFileSync(new URL('../../apps/web/static/shared/playback-worklet.js', import.meta.url), 'utf8'), scope);
  const processor = new Processor({ processorOptions: { sampleRate: 24000 } });
  processor.port.onmessage({ data: { audio: new Int16Array(240002).buffer } });
  assert.equal(messages[0].type, 'overflow');
});
