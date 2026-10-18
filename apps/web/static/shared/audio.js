/** 管理浏览器麦克风；采样率使用 AudioContext 的实际值，交由核心包重采样。 */
export class Microphone {
  async open(deviceId) {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { deviceId: deviceId ? { exact: deviceId } : undefined, channelCount: 1 },
      });
      this.context = new AudioContext();
      await this.context.resume();
      await this.context.audioWorklet.addModule('/static/shared/pcm-worklet.js');
      return this.context.sampleRate;
    } catch (error) {
      await this.dispose();
      throw error;
    }
  }

  start(onAudio, onError, onLevel) {
    this.source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, 'speechio-pcm');
    this.node.port.onmessage = ({ data }) => {
      if (data.type === 'stopped') {
        this.stopped?.();
        return;
      }
      try {
        onAudio(data.bytes);
        const samples = new Int16Array(data.bytes);
        let energy = 0;
        for (const sample of samples) energy += (sample / 32768) ** 2;
        onLevel(Math.min(1, Math.sqrt(energy / samples.length) * 6));
      } catch (error) {
        onError(error);
      }
    };
    this.node.onprocessorerror = () => onError(new Error('浏览器音频处理线程失败'));
    this.stream.getAudioTracks().forEach(track => {
      track.onended = () => onError(new Error('麦克风已断开，请重新选择输入设备'));
    });
    this.source.connect(this.node);
    // 处理器输出静音，连接 destination 保持处理线程运行，不回放麦克风声音。
    this.node.connect(this.context.destination);
  }

  async stop() {
    if (this.stopping) return this.stopping;
    this.stopping = (async () => {
      try {
        if (this.node) {
          if (this.context.state === 'suspended') await this.context.resume();
          await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('麦克风尾部音频交付超时')), 3000);
            this.stopped = () => { clearTimeout(timer); resolve(); };
            this.node.port.postMessage('stop');
          });
        }
      } finally {
        await this.dispose();
      }
    })();
    return this.stopping;
  }

  async dispose() {
    this.stream?.getTracks().forEach(track => track.stop());
    this.source?.disconnect();
    this.node?.disconnect();
    if (this.context && this.context.state !== 'closed') await this.context.close();
  }
}
