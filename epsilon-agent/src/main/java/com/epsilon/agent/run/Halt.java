package com.epsilon.agent.run;

import java.util.Objects;

/**
 * セーフポイントで止まると決めたこと。
 *
 * @param ending 呼び出しの終わり方（止める・起動し直す・GPU を譲る）
 * @param request 従った要求の番号。シグナル・GPU の要求で止まるときは null
 * @param reason 理由（英語の 1 行。status.json とログに書く）
 */
public record Halt(Ending ending, Integer request, String reason) {

  public Halt {
    Objects.requireNonNull(ending, "ending");
    Objects.requireNonNull(reason, "reason");
    if (ending != Ending.STOPPED && ending != Ending.RESTART && ending != Ending.YIELD) {
      throw new IllegalArgumentException("A safe point stops, restarts or yields, not " + ending);
    }
  }
}
