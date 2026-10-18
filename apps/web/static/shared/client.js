/** 两个页面共用的通信客户端；每个实例只拥有自己的连接。 */
export class SpeechClient {
  constructor(onMessage, onClose) {
    this.onMessage = onMessage;
    this.onClose = onClose;
    this.socket = null;
  }

  async start(config, fileToken, debug) {
    const url = new URL('/api/session', location.href);
    url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    if (debug) url.searchParams.set('debug', '1');
    const socket = new WebSocket(url);
    this.socket = socket;
    return new Promise((resolve, reject) => {
      socket.onopen = () => socket.send(JSON.stringify({
        type: 'start', config, file_token: fileToken,
      }));
      socket.onmessage = ({ data }) => {
        const message = JSON.parse(data);
        if (message.type === 'ready') resolve();
        if (message.type === 'error' && message.fatal) reject(new Error(message.message));
        this.onMessage(message);
      };
      socket.onerror = () => reject(new Error('无法连接本机服务，请检查服务是否已启动'));
      socket.onclose = () => {
        reject(new Error('连接已关闭'));
        this.onClose();
      };
    });
  }

  command(type) {
    if (this.socket?.readyState !== WebSocket.OPEN) throw new Error('连接未建立');
    this.socket.send(JSON.stringify({ type }));
  }

  audio(bytes) {
    if (this.socket?.readyState !== WebSocket.OPEN) throw new Error('音频连接已断开');
    // 超过缓冲限制时明确终止，不能悄悄丢弃后续音频。
    if (this.socket.bufferedAmount > 2 * 1024 * 1024) {
      throw new Error('音频发送速度跟不上录音，请结束后重试');
    }
    this.socket.send(bytes);
  }

  close() {
    if (this.socket?.readyState === WebSocket.OPEN) this.command('close');
  }
}

export async function request(path, options = {}) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.message || value.detail || `请求失败：${response.status}`);
  return value;
}
