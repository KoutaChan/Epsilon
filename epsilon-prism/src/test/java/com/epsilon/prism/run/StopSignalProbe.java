package com.epsilon.prism.run;

/**
 * {@link StopSignalTest} の子の JVM: SIGTERM の合図を登録して ready を書き、合図が立つまで回る。立ったら今の段を終えた体で少し待ち、
 * stopped を書いて 0 で終わる（その間の TERM も合図を立てるだけ）。
 */
public final class StopSignalProbe {
  private StopSignalProbe() {}

  public static void main(String[] arguments) throws InterruptedException {
    StopSignal stop = StopSignal.onTerm();
    System.out.println("ready");
    System.out.flush();
    while (!stop.requested()) Thread.sleep(10);
    Thread.sleep(500);
    System.out.println("stopped");
    System.out.flush();
    System.exit(0);
  }
}
