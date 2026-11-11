import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, mkdirSync, openSync, statSync } from "node:fs";
import path from "node:path";
import type { InvokeReply } from "../../../protocol/bridge.ts";
import type { StdioCommandName } from "../../../protocol/commands.ts";
import type { ErrorBody } from "../../../protocol/errors.ts";
import { ENGINE_ENV } from "../../../protocol/stdio.ts";
import { ENGINE_STOPPED, EngineClient } from "./client.ts";

/**
 * Java に渡す引数。Windows の java.exe は引数を ANSI のコードページで読むので、引数は ASCII だけにし、
 * パスは環境変数で渡す。-cp は作業フォルダー（engine）からの相対パスで、logback の設定は jar の中のファイル名。
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

/** stderr のログの上限（バイト）。起動するときにこれを超えていれば、ログを空にする。 */
const STDERR_LOG_LIMIT = 1024 * 1024;

/** stdin を閉じてから kill するまでに待つ時間（ミリ秒）。 */
const STOP_GRACE_MS = 5000;

/** 子プロセスを起動するための設定。 */
export interface EngineLaunch {
  command: string;
  args: readonly string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  /** stderr を追記していくファイル。 */
  stderrLog: string;
}

/**
 * 同梱のエンジンを起動するための設定を作る。パスは環境変数（ENGINE_ENV と DJL の変数）で渡す。DJL には同梱の
 * libtorch の CPU 版を使わせ、ダウンロードも利用状況の送信もさせない。
 *
 * @param engine エンジンのフォルダー（resources/engine。パッケージしていなければ desktop/engine）
 * @param userData Electron の userData のフォルダー
 * @param resultsDirectory 結果の保存先のフォルダー
 */
export function engineLaunch(
  engine: string,
  userData: string,
  resultsDirectory: string,
): EngineLaunch {
  const logs = path.join(userData, "logs");
  const cache = path.join(userData, "djl");
  const env: NodeJS.ProcessEnv = { ...process.env };
  // 利用者の環境にある JVM のオプションを混ぜない。JVM に渡すのは上の決まった引数だけ。
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
  /** 子プロセスが終わり、応答を待っていた要求を断った後に resolve する。 */
  closed: Promise<void>;
}

/**
 * Java の子プロセスを 1 つ動かす。子プロセスが落ちたら、次の要求で起動し直す。起動の失敗（failed のイベント。
 * data_locked など）を受けたら、restart（保存先の変更）まではどの要求にもそのエラーを返す。restart と stop は
 * 1 つずつ順に行い、要求はそれが済んでから送る。
 */
export class EngineProcess {
  readonly #launch: () => EngineLaunch;
  readonly #stopGraceMs: number;
  #running: Running | null = null;
  #failure: ErrorBody | null = null;
  #stopped = false;
  #transition: Promise<void> = Promise.resolve();

  /** @param launch 起動のたびに呼ぶ関数（その時点の保存先を読む） */
  constructor(launch: () => EngineLaunch, stopGraceMs = STOP_GRACE_MS) {
    this.#launch = launch;
    this.#stopGraceMs = stopGraceMs;
  }

  /** 子プロセスが動いていなければ起動する。起動に失敗した後や stop の後は何もしない。 */
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

  /** 起動に失敗したままか。true の間は、restart するまでどの要求も断る。 */
  get failed(): boolean {
    return this.#failure !== null;
  }

  /** 子プロセスを止め、その時点の設定で起動し直す。覚えていた起動の失敗も忘れる。 */
  restart(): Promise<void> {
    return this.#serialize(async () => {
      await this.#halt();
      this.#failure = null;
      this.start();
    });
  }

  /** アプリを終えるときに呼び、子プロセスを止める。この後の要求は断る。 */
  stop(): Promise<void> {
    this.#stopped = true;
    return this.#serialize(() => this.#halt());
  }

  #serialize(step: () => Promise<void>): Promise<void> {
    const next = this.#transition.then(step);
    this.#transition = next.catch(() => undefined);
    return next;
  }

  /** stdin を閉じて子プロセスが終わるのを待つ。決まった時間を過ぎたら kill する。 */
  async #halt(): Promise<void> {
    const running = this.#running;
    if (running === null) return;
    running.child.stdin?.end();
    const timer = setTimeout(() => running.child.kill(), this.#stopGraceMs);
    await running.closed;
    clearTimeout(timer);
  }

  /** 動いている子プロセスを返す。無ければ起動し、起動できなければ（ログのフォルダーを作れないときなど）null。 */
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
    // 終わった子プロセスへ書き込んで失敗しても、ここでは無視する。後始末は close で行う。
    stdin.on("error", () => undefined);
    child.on("error", (error) =>
      console.error("The analysis engine process failed:", error),
    );
    const client = new EngineClient(stdin, stdout, () => child.kill());
    const running: Running = {
      child,
      client,
      // close のイベントは、起動できなかったとき（ENOENT など）にも来る。
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

/** stderr のログを開き、子プロセスへ渡す fd を返す。ログが上限より大きければ、空にしてから書く。 */
function openStderrLog(file: string): number {
  mkdirSync(path.dirname(file), { recursive: true });
  const size = statSync(file, { throwIfNoEntry: false })?.size ?? 0;
  return openSync(file, size > STDERR_LOG_LIMIT ? "w" : "a");
}
