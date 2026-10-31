/** A few decode workers; urgent jobs (the frame on screen, a thumbnail) jump the queue. */
import type { Decoded, FrameInfo } from "./pixels";

interface Job {
  bytes: Uint8Array;
  info: FrameInfo;
  resolve: (d: Decoded) => void;
  reject: (e: Error) => void;
}

class DecodePool {
  private workers: Worker[] = [];
  private idle: Worker[] = [];
  private queue: Job[] = [];
  private pending = new Map<number, Job>();
  private nextId = 1;
  readonly size: number;

  constructor() {
    // iPad: 2–4 workers are plenty and each holds its own WASM heap
    this.size = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 4) - 1));
  }

  private worker(): Worker | null {
    if (this.idle.length) return this.idle.pop()!;
    if (this.workers.length >= this.size) return null;
    const w = new Worker(new URL("./decode.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e) => {
      const { id, gray, rgb, error } = e.data;
      const job = this.pending.get(id);
      this.pending.delete(id);
      this.idle.push(w);
      if (job) {
        if (error) job.reject(new Error(error));
        else job.resolve({ gray, rgb });
      }
      this.pump();
    };
    this.workers.push(w);
    return w;
  }

  private pump() {
    while (this.queue.length) {
      const w = this.worker();
      if (!w) return;
      const job = this.queue.shift()!;
      const id = this.nextId++;
      this.pending.set(id, job);
      w.postMessage({ id, bytes: job.bytes, info: job.info }, [job.bytes.buffer]);
    }
  }

  /** Decode one frame in a worker. `bytes` is transferred (the caller must not use it afterwards). */
  decode(bytes: Uint8Array, info: FrameInfo, urgent = false): Promise<Decoded> {
    return new Promise((resolve, reject) => {
      const job = { bytes, info, resolve, reject };
      if (urgent) this.queue.unshift(job);
      else this.queue.push(job);
      this.pump();
    });
  }

  get busy(): number {
    return this.pending.size + this.queue.length;
  }
}

let pool: DecodePool | null = null;
export function decodePool(): DecodePool {
  return (pool ??= new DecodePool());
}
