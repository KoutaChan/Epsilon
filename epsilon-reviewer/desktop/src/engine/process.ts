import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, mkdirSync, openSync, statSync } from "node:fs";
import path from "node:path";
import type { InvokeReply } from "../../../protocol/bridge.ts";
import type { StdioCommandName } from "../../../protocol/commands.ts";
import type { ErrorBody } from "../../../protocol/errors.ts";
import { ENGINE_ENV } from "../../../protocol/stdio.ts";
import { ENGINE_STOPPED, EngineClient } from "./client.ts";

/**
 * Java の引数。Windows の java.exe は引数を ANSI の符号で読むので ASCII だけにし、パスは環境変数で渡す。
 * -cp は作業のフォルダー（engine）からの相対、logback の設定は jar の中の名前。
 */
const JAVA_ARGUMENTS = [
  "-Xms32m",
  "-Xmx1g",
  "-XX:+UseSerialGC",
  "-Dfile.encoding=UTF-8",
  "-Dlogback.configurationFile=reviewer-stdio-logback.xml",
  "-cp",
  "lib\\epsilon-reviewer.jar",
  "com.epsilon.reviewer.StdioMain",
] as const;

/** 起こす時に stderr の記録がこれを超えていれば空にする（バイト）。 */
const STDERR_LOG_LIMIT = 1024 * 1024;

/** stdin を閉じてから kill するまでの待ち（ミリ秒）。 */
const STOP_GRACE_MS = 5000;

/** 子プロセスの起こし方。 */
export interface EngineLaunch {
  command: string;
  args: readonly string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  /** stderr を足していくファイル。 */
  stderrLog: string;
}

/**
 * 同梱のエンジンの起こし方。パスは環境変数（ENGINE_ENV と DJL の変数）で渡す。DJL は同梱の libtorch の CPU 版を
 * 使い、ダウンロードと利用の記録の送信をしない。
 *
 * @param engine resources/engine（未梱包は desktop/engine）
 * @param userData Electron の userData
 * @param resultsDirectory 保存先のフォルダー
 */
export function engineLaunch(
  engine: string,
  userData: string,
  resultsDirectory: string,
): EngineLaunch {
  const logs = path.join(userData, "logs");
  const cache = path.join(userData, "djl");
  const env: NodeJS.ProcessEnv = { ...process.env };
  // 利用者の環境の JVM の選択肢を混ぜない（引数は上の固定のものだけ）。
  delete env.JAVA_TOOL_OPTIONS;
  delete env._JAVA_OPTIONS;
  delete env.JDK_JAVA_OPTIONS;
  Object.assign(env, {
    [ENGINE_ENV.data]: resultsDirectory,
    [ENGINE_ENV.models]: path.join(engine, "models", "models.json"),
    [ENGINE_ENV.logs]: logs,
    PYTORCH_LIBRARY_PATH: path.join(engine, "native"),
    PYTORCH_FLAVOR: "cpu",
    PYTORCH_VERSION: "2.11.0",
    DJL_OFFLINE: "true",
    OPT_OUT_TRACKING: "true",
    ENGINE_CACHE_DIR: cache,
    DJL_CACHE_DIR: cache,
    OMP_NUM_THREADS: "2",
    KMP_BLOCKTIME: "1",
  });
  return {
    command: path.join(engine, "runtime", "bin", "java.exe"),
    args: JAVA_ARGUMENTS,
    cwd: engine,
    env,
    stderrLog: path.join(logs, "engine-stderr.log"),
  };
}

interface Running {
  child: ChildProcess;
  client: EngineClient;
  /** 子プロセスが終わり、待ちの要求を断った後に解ける。 */
  closed: Promise<void>;
}

/**
 * Java の子プロセス 1 つ。落ちれば次の要求で起こし直す。起動の失敗（failed の知らせ、例えば data_locked）は
 * restart（保存先の変更）まで答え続ける。restart と stop は順に 1 つずつ行い、要求はその後に送る。
 */
export class EngineProcess {
  readonly #launch: () => EngineLaunch;
  readonly #stopGraceMs: number;
  #running: Running | null = null;
  #failure: ErrorBody | null = null;
  #stopped = false;
  #transition: Promise<void> = Promise.resolve();

  /** @param launch 起こす度に呼ぶ（その時の保存先を読む） */
  constructor(launch: () => EngineLaunch, stopGraceMs = STOP_GRACE_MS) {
    this.#launch = launch;
    this.#stopGraceMs = stopGraceMs;
  }

  /** 動いていなければ起こす。 */
  start(): void {
    if (this.#failure === null && !this.#stopped) this.#ensure();
  }

  async request(
    command: StdioCommandName,
    request: object,
    body?: ArrayBuffer,
  ): Promise<InvokeReply> {
    await this.#transition;
    const refusal = this.#stopped ? ENGINE_STOPPED : this.#failure;
    if (refusal !== null) return { ok: false, error: refusal };
    const running = this.#ensure();
    return running === null
      ? { ok: false, error: ENGINE_STOPPED }
      : running.client.request(command, request, body);
  }

  /** 止めて、その時の設定で起こし直す。起動の失敗も忘れる。 */
  restart(): Promise<void> {
    return this.#serialize(async () => {
      await this.#halt();
      this.#failure = null;
      this.start();
    });
  }

  /** アプリの終わり。以後の要求は断る。 */
  stop(): Promise<void> {
    this.#stopped = true;
    return this.#serialize(() => this.#halt());
  }

  #serialize(step: () => Promise<void>): Promise<void> {
    const next = this.#transition.then(step);
    this.#transition = next.catch(() => undefined);
    return next;
  }

  /** stdin を閉じ、終わりを待つ。待ちきれなければ kill する。 */
  async #halt(): Promise<void> {
    const running = this.#running;
    if (running === null) return;
    running.child.stdin?.end();
    const timer = setTimeout(() => running.child.kill(), this.#stopGraceMs);
    await running.closed;
    clearTimeout(timer);
  }

  /** 動いている子プロセス。起こせなければ（記録の場所が作れないなど）null。 */
  #ensure(): Running | null {
    if (this.#running === null) {
      try {
        this.#running = this.#spawn();
      } catch (error) {
        console.error("The analysis engine could not start:", error);
      }
    }
    return this.#running;
  }

  #spawn(): Running {
    const launch = this.#launch();
    const log = openStderrLog(launch.stderrLog);
    let child: ChildProcess;
    try {
      child = spawn(launch.command, launch.args, {
        cwd: launch.cwd,
        env: launch.env,
        windowsHide: true,
        stdio: ["pipe", "pipe", log],
      });
    } finally {
      closeSync(log);
    }
    const { stdin, stdout } = child;
    if (stdin === null || stdout === null)
      throw new Error("The engine process has no stdio pipes.");
    // 終わった子プロセスへの書き込みの失敗は close で扱う。
    stdin.on("error", () => undefined);
    child.on("error", (error) =>
      console.error("The analysis engine process failed:", error),
    );
    const client = new EngineClient(stdin, stdout, () => child.kill());
    const running: Running = {
      child,
      client,
      // close は起こせなかった時（ENOENT など）も出る。
      closed: new Promise((resolve) => {
        child.once("close", () => {
          client.close();
          if (client.failure !== null) this.#failure = client.failure;
          if (this.#running === running) this.#running = null;
          resolve();
        });
      }),
    };
    return running;
  }
}

/** stderr の記録を開いて子プロセスへ渡す fd を返す。大きくなっていれば空にしてから書く。 */
function openStderrLog(file: string): number {
  mkdirSync(path.dirname(file), { recursive: true });
  const size = statSync(file, { throwIfNoEntry: false })?.size ?? 0;
  return openSync(file, size > STDERR_LOG_LIMIT ? "w" : "a");
}
