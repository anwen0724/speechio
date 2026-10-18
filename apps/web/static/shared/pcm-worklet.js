/** 音频线程：把浏览器浮点采样转为单声道 int16，结束时交付不足一块的尾部。 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Int16Array(1024);
    this.length = 0;
    this.running = true;
    this.port.onmessage = ({ data }) => {
      if (data === 'stop') {
        this.running = false;
        this.flush();
        this.port.postMessage({ type: 'stopped' });
      }
    };
  }

  flush() {
    if (!this.length) return;
    const bytes = new ArrayBuffer(this.length * 2);
    const view = new DataView(bytes);
    for (let i = 0; i < this.length; i++) view.setInt16(i * 2, this.buffer[i], true);
    this.port.postMessage({ type: 'audio', bytes }, [bytes]);
    this.length = 0;
  }

  process(inputs) {
    if (!this.running) return false;
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i];
      value = Math.max(-1, Math.min(1, value / channels.length));
      this.buffer[this.length++] = Math.round(value * (value < 0 ? 32768 : 32767));
      if (this.length === this.buffer.length) this.flush();
    }
    return true;
  }
}

registerProcessor('speechio-pcm', PCMProcessor);
