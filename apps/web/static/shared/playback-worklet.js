/** 在浏览器音频线程按顺序播放 PCM；采样率不同时做线性插值。 */
class SpeechPlayback extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.rate = options.processorOptions.sampleRate;
    this.buffer = new Float32Array(this.rate * 10 + 1);
    this.readIndex = 0;
    this.length = 0;
    this.position = 0;
    this.consumed = 0;
    this.finished = false;
    this.port.onmessage = ({ data }) => {
      if (data.type === 'finish') {
        this.finished = true;
        return;
      }
      const view = new DataView(data.audio);
      const count = view.byteLength / 2;
      if (count > this.buffer.length - this.length) {
        this.port.postMessage({ type: 'overflow' });
        return;
      }
      for (let i = 0; i < count; i++) {
        this.buffer[(this.readIndex + this.length + i) % this.buffer.length] = view.getInt16(i * 2, true) / 32768;
      }
      this.length += count;
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0];
    const ratio = this.rate / sampleRate;
    for (let i = 0; i < output.length; i++) {
      if (this.length === 0) break;
      const first = this.buffer[this.readIndex];
      const next = this.length > 1 ? this.buffer[(this.readIndex + 1) % this.buffer.length] : first;
      output[i] = first + (next - first) * this.position;
      this.position += ratio;
      const advance = Math.min(Math.floor(this.position), this.length);
      this.position -= Math.floor(this.position);
      this.readIndex = (this.readIndex + advance) % this.buffer.length;
      this.length -= advance;
      this.consumed += advance;
    }
    if (this.consumed > 0 && (this.consumed >= 1024 || this.length === 0)) {
      this.port.postMessage({ type: 'consumed', samples: this.consumed });
      this.consumed = 0;
    }
    if (this.finished && this.length === 0) {
      this.port.postMessage({ type: 'ended' });
      return false;
    }
    return true;
  }
}

registerProcessor('speechio-playback', SpeechPlayback);
