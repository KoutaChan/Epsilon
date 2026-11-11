package com.epsilon.agent.run;

/**
 * 段の中で待っている間に、セーフポイントで止まると決めた（{@link RunControl#await}）。段は完了マーカーを置かずに抜け、
 * {@link Pipeline} がこれを受けて反復を止める。
 */
public final class Halted extends Exception {

  private final Halt halt;

  public Halted(Halt halt) {
    super(halt.reason(), null, false, false);
    this.halt = halt;
  }

  public Halt halt() {
    return halt;
  }
}
