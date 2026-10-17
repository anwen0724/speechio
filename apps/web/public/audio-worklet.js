/** 音频线程只合并源帧，不做重采样；短尾必须先于 flushed 发送。 */
class SourceFrames extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = false;
    this.block = new Float32Array(1024);
    this.length = 0;
    this.port.onmessage = ({ data }) => {
      if (data === 'start') this.active = true;
      if (data === 'flush') {
        this.active = false;
        if (this.length) this.emit();
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }

  emit() {
    const frames = this.block.slice(0, this.length);
    this.port.postMessage({ type: 'audio', frames }, [frames.buffer]);
    this.length = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (this.active && channel) {
      for (const sample of channel) {
        this.block[this.length++] = sample;
        if (this.length === this.block.length) this.emit();
      }
    }
    return true;
  }
}

registerProcessor('source-frames', SourceFrames);
