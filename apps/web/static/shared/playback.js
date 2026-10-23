/** 只负责当前任务的试听；停止试听不会向合成服务发送取消请求。 */
export class AudioPlayback {
  constructor(onState) {
    this.onState = onState;
    this.context = null;
    this.node = null;
    this.pending = 0;
  }

  async start(rate) {
    await this.stop();
    this.rate = rate;
    this.context = new AudioContext();
    await this.context.resume();
    await this.context.audioWorklet.addModule('/static/shared/playback-worklet.js');
    this.node = new AudioWorkletNode(this.context, 'speechio-playback', {
      numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1],
      processorOptions: { sampleRate: rate },
    });
    this.node.port.onmessage = ({ data }) => {
      if (data.type === 'consumed') this.pending = Math.max(0, this.pending - data.samples);
      if (data.type === 'ended') void this.stop('试听已结束');
      if (data.type === 'overflow') void this.stop('试听跟不上生成速度，完成后可播放完整音频');
    };
    this.node.connect(this.context.destination);
    this.onState('等待音频');
  }

  push(buffer) {
    if (!this.node) return;
    this.pending += buffer.byteLength / 2;
    if (this.pending > this.rate * 10) {
      void this.stop('试听积压超过 10 秒，完成后可播放完整音频');
      return;
    }
    this.node.port.postMessage({ audio: buffer }, [buffer]);
    this.onState('正在试听');
  }

  finish() {
    this.node?.port.postMessage({ type: 'finish' });
  }

  async stop(message = '试听已停止') {
    const context = this.context;
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node.port.close();
    }
    this.node = this.context = null;
    this.pending = 0;
    if (context && context.state !== 'closed') await context.close();
    this.onState(message);
  }
}
