package com.epsilon.agent.app;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.Signal;

/**
 * SIGTERM と SIGINT のハンドラ。コマンドが停止の要求を受けると宣言すれば（{@link Context#signals}）、どちらも次のセーフポイントで
 * 止める要求になる。そのときも、2 回目の SIGINT ではすぐに終わる。宣言していなければ、最初の割り込みですぐに終わる。
 */
public final class Signals {

  private static final Logger log = LoggerFactory.getLogger(Signals.class);

  private final AtomicInteger interrupts = new AtomicInteger();
  private volatile boolean claimed;
  private volatile String request;

  Signals() {}

  /** ハンドラを入れる。プロセスで 1 回だけ、main の最初に呼ぶ。 */
  void install(IntConsumer exit) {
    Signal.handle(new Signal("TERM"), signal -> receive("SIGTERM", false, exit));
    Signal.handle(new Signal("INT"), signal -> receive("SIGINT", true, exit));
  }

  /** 停止の要求をセーフポイントで読むと宣言する。 */
  void claim() {
    claimed = true;
  }

  /** 停止の要求を置く。2 回目からは何もしない。 */
  public void request(String reason) {
    if (request == null) {
      request = reason;
      log.info("{}: stopping at the next safe point", reason);
    }
  }

  /** 停止の要求があるか。 */
  public boolean requested() {
    return request != null;
  }

  /** @return 停止の要求の理由（SIGTERM など）。要求が無ければ null */
  public String reason() {
    return request;
  }

  private void receive(String name, boolean interrupt, IntConsumer exit) {
    boolean again = interrupt && interrupts.incrementAndGet() > 1;
    if (!claimed || again) {
      log.warn("{}: exiting without waiting for a safe point", name);
      exit.accept(Exit.INTERRUPTED.code());
      return;
    }
    request(name);
  }
}
