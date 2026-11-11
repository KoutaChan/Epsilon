package com.epsilon.agent.run;

import java.time.Instant;
import java.util.Objects;

/**
 * control に置く要求のファイル（{@code <番号>-<種類>.json}）の中身。
 *
 * @param number 要求の番号（1 から。control と control/done を合わせて重ならない）
 * @param time 要求を出した時刻
 * @param request 要求
 */
public record ControlRequest(int number, Instant time, Request request) {

  public ControlRequest {
    if (number < 1) {
      throw new IllegalArgumentException("A request number starts at 1: " + number);
    }
    Objects.requireNonNull(time, "time");
    Objects.requireNonNull(request, "request");
  }

  /** ファイル名。 */
  public String fileName() {
    return number + "-" + request.label() + ".json";
  }
}
