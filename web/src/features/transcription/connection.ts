import { z } from 'zod';

const unitSchema = z.object({
  session_id: z.string(),
  generation: z.number().int().nonnegative(),
  unit_id: z.number().int().positive(),
});
export type UnitRef = z.infer<typeof unitSchema>;
const candidateRequestSchema = z.object({
  unit: unitSchema,
  request_id: z.string(),
  kind: z.enum(['tail', 'whole', 'correction']),
  hint: z.boolean(),
  start_sample: z.number().int(),
  end_sample: z.number().int(),
  audio_version: z.number().int(),
  resume_epoch: z.number().int(),
  resume_id: z.string().nullable(),
  candidate_start_sample: z.number().int(),
});
const candidateSchema = z
  .object({
    request: candidateRequestSchema,
    text: z.string(),
    live_text: z.string(),
    model_text: z.string(),
    status: z.string(),
    current: z.boolean(),
    matched: z.boolean(),
    used_fuzzy_match: z.boolean(),
    attempts: z.number().int(),
    added_silence_samples: z.number().int(),
    replaces_task_id: z.string().nullable(),
  })
  .passthrough();
export type CandidateResult = z.infer<typeof candidateSchema>;
export type CompletionMode = 'final' | 'candidate';
type OpenOptions = {
  stableText: boolean;
  dualRealtime: boolean;
  completionMode: CompletionMode;
  retainForCorrection: boolean;
};
const mergeSchema = z
  .object({
    unit: unitSchema,
    text: z.string(),
    accurate_text: z.string(),
    ultrafast_text: z.string(),
    ultrafast_suffix: z.string(),
    status: z.enum([
      'waiting_for_accurate',
      'accurate_only',
      'exact',
      'bootstrap_exact',
      'fuzzy',
      'soft_frontier',
      'held_fuzzy_pending',
      'held_no_anchor',
      'held_fast_conflict',
      'accurate_advanced_no_anchor',
      'stale_ignored',
    ]),
    revision: z.number().int().nonnegative(),
    accurate_generation: z.number().int().nonnegative(),
    matched: z.boolean(),
    held: z.boolean(),
  })
  .passthrough();
export type RealtimeMerge = z.infer<typeof mergeSchema>;
const eventSchema = z
  .object({
    kind: z.enum([
      'realtime',
      'final',
      'completed',
      'failed',
      'realtime_failed',
      'cancelled',
      'candidate',
      'candidate_failed',
      'correction',
      'correction_failed',
    ]),
    unit: unitSchema,
    sequence: z.number().int(),
    task_id: z.string().nullable().optional(),
    resume_epoch: z.number().int().optional(),
    candidate: candidateSchema.nullable().optional(),
    result: z.object({ text: z.string() }).passthrough().nullable(),
    error: z.object({ code: z.string(), message: z.string() }).nullable(),
    realtime_role: z.enum(['accurate', 'ultrafast']).nullable().optional(),
    merge: mergeSchema.nullable().optional(),
    stabilization: z
      .object({
        accepted: z.boolean(),
        stable_text: z.string(),
        unstable_text: z.string(),
        display_text: z.string(),
        stable_prefix_conflict: z.boolean(),
      })
      .passthrough()
      .nullable()
      .optional(),
    finalization: z
      .object({
        agrees_with_stable_prefix: z.boolean(),
        mismatch_reason: z.string().nullable(),
      })
      .passthrough()
      .nullable()
      .optional(),
  })
  .passthrough();
const messageSchema = z
  .object({
    protocol_version: z.literal(1),
    type: z.enum(['ready', 'ack', 'pong', 'event', 'error', 'closed']),
    session_id: z.string().optional(),
    generation: z.number().int().nonnegative().optional(),
    request_id: z.string().nullable().optional(),
    command: z.string().optional(),
    unit: unitSchema.optional(),
    candidate: z.object({ enabled: z.boolean() }).passthrough().optional(),
    resume: z
      .object({
        unit: unitSchema,
        resume_epoch: z.number().int(),
        resume_id: z.string(),
        candidate_start_sample: z.number().int(),
      })
      .optional(),
    event: eventSchema.optional(),
    code: z.string().optional(),
    message: z.string().optional(),
  })
  .passthrough();
const workletMessageSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('audio'), frames: z.instanceof(Float32Array) }),
  z.object({ type: z.literal('flushed') }),
]);
export type WireMessage = z.infer<typeof messageSchema>;
export type Mode = 'manual' | 'auto';
type Pending = {
  resolve: (message: WireMessage) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

/** 每次采集拥有设备流和音频图；停止后可以在相同源格式下重开。 */
class Capture {
  readonly context: AudioContext;
  readonly stream: MediaStream;
  readonly source: MediaStreamAudioSourceNode;
  readonly node: AudioWorkletNode;
  readonly mute: GainNode;
  private flushWait:
    | {
        resolve: () => void;
        reject: (error: Error) => void;
        timer: ReturnType<typeof setTimeout>;
      }
    | undefined;

  private constructor(
    context: AudioContext,
    stream: MediaStream,
    node: AudioWorkletNode,
    onAudio: (frames: Float32Array) => void,
    onError: (error: Error) => void,
  ) {
    this.context = context;
    this.stream = stream;
    this.node = node;
    this.source = context.createMediaStreamSource(stream);
    this.mute = context.createGain();
    this.mute.gain.value = 0;
    this.source.connect(node).connect(this.mute).connect(context.destination);
    node.port.onmessage = ({ data }: MessageEvent<unknown>) => {
      try {
        const message = workletMessageSchema.parse(data);
        if (message.type === 'flushed') {
          const waiting = this.flushWait;
          this.flushWait = undefined;
          if (waiting) {
            clearTimeout(waiting.timer);
            waiting.resolve();
          }
        } else onAudio(message.frames);
      } catch (error) {
        onError(asError(error));
      }
    };
    node.onprocessorerror = () => onError(new Error('AudioWorklet 处理失败'));
  }

  static async create(
    sampleRate: number | undefined,
    onAudio: (frames: Float32Array) => void,
    onError: (error: Error) => void,
  ) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    let context: AudioContext | undefined;
    try {
      context = new AudioContext(sampleRate ? { sampleRate } : undefined);
      await context.audioWorklet.addModule('/audio-worklet.js');
      const node = new AudioWorkletNode(context, 'source-frames', {
        channelCount: 1,
        channelCountMode: 'explicit',
      });
      return new Capture(context, stream, node, onAudio, onError);
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      if (context) await context.close();
      throw error;
    }
  }

  async start() {
    await this.context.resume();
    this.node.port.postMessage('start');
  }

  async flush() {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.flushWait = undefined;
        reject(new Error('音频短尾刷新超时'));
      }, 5000);
      this.flushWait = { resolve, reject, timer };
      this.node.port.postMessage('flush');
    });
  }

  async release(reason = new Error('音频采集已关闭')) {
    const waiting = this.flushWait;
    this.flushWait = undefined;
    if (waiting) {
      clearTimeout(waiting.timer);
      waiting.reject(reason);
    }
    this.node.port.onmessage = null;
    this.node.onprocessorerror = null;
    this.node.port.close();
    this.source.disconnect();
    this.node.disconnect();
    this.mute.disconnect();
    this.stream.getTracks().forEach((track) => track.stop());
    if (this.context.state !== 'closed') await this.context.close();
  }
}

/** 封装协议归属、请求交接和全部浏览器资源，不让 React 每帧参与采集。 */
export class BrowserSession {
  private socket: WebSocket | undefined;
  private capture: Capture | undefined;
  private sampleRate = 0;
  private sessionId = '';
  private generation = 0;
  private unitId: number | null = null;
  private sequence = 0;
  private pending = new Map<string, Pending>();
  private disposed = false;
  private receivingClosed = false;
  private capturing = false;
  private inputEnded = false;
  private ready: Pending | undefined;
  private resumeEpoch = 0;

  constructor(
    readonly mode: Mode,
    private readonly onMessage: (message: WireMessage) => void,
    private readonly onError: (error: Error) => void,
  ) {}

  /** 先取得实际设备格式，再以 open 鉴权；成功连接尚未开始递交音频。 */
  async connect(token: string, options: OpenOptions) {
    try {
      this.capture = await Capture.create(
        undefined,
        (frames) => this.sendAudio(frames),
        (error) => this.fail(error),
      );
      this.sampleRate = this.capture.context.sampleRate;
      if (this.disposed) {
        await this.capture.release();
        this.capture = undefined;
        return;
      }
      const url = new URL('/api/v1/ws', window.location.href);
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(url);
      this.socket = socket;
      const ready = new Promise<WireMessage>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('等待会话就绪超时')), 10000);
        this.ready = { resolve, reject, timer };
      });
      socket.onopen = () =>
        socket.send(
          JSON.stringify({
            protocol_version: 1,
            type: 'open',
            request_id: crypto.randomUUID(),
            token: token || null,
            mode: this.mode,
            realtime: true,
            ...(options.stableText ? { stable_text: true } : {}),
            ...(options.dualRealtime ? { dual_realtime: true } : {}),
            completion_mode: options.completionMode,
            retain_for_correction: options.retainForCorrection,
            format: { sample_rate: this.sampleRate, channels: 1, sample_type: 'float32' },
          }),
        );
      socket.onmessage = ({ data }: MessageEvent<unknown>) => this.receive(data);
      socket.onerror = () => this.fail(new Error('WebSocket 连接失败'));
      socket.onclose = () => {
        if (!this.disposed && !this.receivingClosed) this.fail(new Error('连接在正常结束前断开'));
      };
      const message = await ready;
      if (message.session_id === undefined || message.generation === undefined)
        throw new Error('ready 缺少会话归属');
      this.sessionId = message.session_id;
      this.generation = message.generation;
    } catch (error) {
      await this.dispose();
      throw error;
    }
  }

  private receive(data: unknown) {
    if (this.disposed) return;
    try {
      if (typeof data !== 'string') throw new Error('服务响应不是 JSON 文本');
      const message = messageSchema.parse(JSON.parse(data));
      // 应答先建立继续边界；已经发送到网络的旧观察也不能恢复旧预览。
      if (message.resume) this.resumeEpoch = message.resume.resume_epoch;
      if (
        message.event?.unit.unit_id === this.unitId &&
        (message.event.resume_epoch ?? 0) < this.resumeEpoch
      )
        return;
      if (message.type === 'ready' && this.ready) {
        clearTimeout(this.ready.timer);
        this.ready.resolve(message);
        this.ready = undefined;
      }
      const pending = message.request_id ? this.pending.get(message.request_id) : undefined;
      if (pending) {
        clearTimeout(pending.timer);
        this.pending.delete(message.request_id ?? '');
        if (message.type === 'error')
          pending.reject(new Error(`${message.code}: ${message.message}`));
        else pending.resolve(message);
      }
      if (message.type === 'error' && !pending) {
        this.fail(new Error(`${message.code}: ${message.message}`));
        return;
      }
      if (message.type === 'closed') this.receivingClosed = true;
      this.onMessage(message);
    } catch (error) {
      this.fail(asError(error));
    }
  }

  private command(type: string, fields: Record<string, unknown> = {}) {
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN || this.disposed)
      return Promise.reject(new Error('会话没有连接'));
    const requestId = crypto.randomUUID();
    return new Promise<WireMessage>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`${type} 应答超时`));
      }, 30000);
      this.pending.set(requestId, { resolve, reject, timer });
      socket.send(
        JSON.stringify({
          protocol_version: 1,
          type,
          request_id: requestId,
          session_id: this.sessionId,
          generation: this.generation,
          ...fields,
        }),
      );
    });
  }

  /** 手动建立单元；自动 EOF 后先重置，再启用采集。 */
  async start() {
    // 自动源在上次 EOF 后建立新代次，不能继续已刷新的转换器。
    if (this.mode === 'auto' && this.inputEnded) await this.reset();
    if (!this.capture)
      this.capture = await Capture.create(
        this.sampleRate,
        (frames) => this.sendAudio(frames),
        (error) => this.fail(error),
      );
    if (this.disposed) {
      await this.capture.release();
      this.capture = undefined;
      throw new Error('会话已经关闭');
    }
    if (this.mode === 'manual') {
      const message = await this.command('start');
      if (!message.unit) throw new Error('start 缺少音频单元');
      this.unitId = message.unit.unit_id;
      this.resumeEpoch = 0;
      this.sequence = 0;
    }
    this.capturing = true;
    await this.capture.start();
  }

  private sendAudio(frames: Float32Array) {
    const socket = this.socket;
    if (!this.capturing || !socket || this.disposed) return;
    if (socket.bufferedAmount > 1048576) throw new Error('网络发送积压，采集已停止');
    const metadata = new TextEncoder().encode(
      JSON.stringify({
        protocol_version: 1,
        session_id: this.sessionId,
        generation: this.generation,
        unit_id: this.unitId,
        audio_sequence: this.sequence,
        frame_count: frames.length,
        format: { sample_rate: this.sampleRate, channels: 1, sample_type: 'float32' },
      }),
    );
    const packet = new ArrayBuffer(4 + metadata.length + frames.byteLength);
    new DataView(packet).setUint32(0, metadata.length, true);
    new Uint8Array(packet, 4, metadata.length).set(metadata);
    const view = new DataView(packet);
    for (let i = 0; i < frames.length; i++)
      view.setFloat32(4 + metadata.length + i * 4, frames[i], true);
    socket.send(packet);
    this.sequence++;
  }

  private async stopCapture(flush: boolean, reason = new Error('音频采集已停止')) {
    if (this.capture) {
      if (flush && this.capturing) await this.capture.flush();
      this.capturing = false;
      await this.capture.release(reason);
      this.capture = undefined;
    }
  }

  /** 先刷新并释放设备，再封口；自动模式结束整个持续源。 */
  async finish() {
    await this.stopCapture(true);
    await this.command(this.mode === 'auto' ? 'end_input' : 'finish');
    if (this.mode === 'auto') this.inputEnded = true;
    this.unitId = null;
  }

  /** 停止采集并舍弃尚未递交的设备尾部，再取消当前录音段。 */
  async cancel() {
    await this.stopCapture(false);
    await this.command('cancel');
    this.unitId = null;
  }

  /** 释放本地采集后请求新代次，旧结果展示由页面另行维护。 */
  async reset() {
    await this.stopCapture(false);
    const message = await this.command('reset');
    if (message.generation === undefined) throw new Error('reset 缺少代次');
    this.generation = message.generation;
    this.inputEnded = false;
    this.sequence = 0;
    this.unitId = null;
    this.resumeEpoch = 0;
  }

  /** 候选控制只递交请求，不停止当前采集。 */
  requestCandidate(kind: 'tail' | 'whole', hint: boolean) {
    return this.command('candidate', { candidate_kind: kind, hint });
  }

  /** 刷新设备短尾，再让服务返回实际标准音频边界。 */
  async resume() {
    if (this.capture && this.capturing) await this.capture.flush();
    return this.command('resume');
  }

  requestCorrection(unit: UnitRef) {
    return this.command('correct', { unit });
  }

  /** 请求正常排空，收到 closed 前继续接收本连接的最终消息。 */
  async close() {
    await this.stopCapture(true);
    await this.command('close');
    // closed 之后才断开，保留排空中的最终消息。
  }

  private fail(error: Error) {
    if (this.disposed) return;
    this.onError(error);
    this.dispose(error).catch((cleanup) => this.onError(asError(cleanup)));
  }

  /** 立即释放自有资源和等待器；失败关闭保留原始原因。 */
  async dispose(reason = new Error('连接已关闭')) {
    if (this.disposed) return;
    this.disposed = true;
    this.capturing = false;
    if (this.ready) {
      clearTimeout(this.ready.timer);
      this.ready.reject(reason);
      this.ready = undefined;
    }
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(reason);
    }
    this.pending.clear();
    if (this.socket) {
      this.socket.onopen = this.socket.onmessage = this.socket.onerror = this.socket.onclose = null;
      this.socket.close();
    }
    await this.stopCapture(false, reason);
  }
}

export function asError(error: unknown) {
  return error instanceof Error ? error : new Error(String(error));
}
