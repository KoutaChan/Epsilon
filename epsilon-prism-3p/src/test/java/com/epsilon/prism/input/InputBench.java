package com.epsilon.prism.input;

import com.epsilon.mahjong.rules.Variant;

/** 3 人打ちの観測の取り込みと入力の符号化を測る。 */
public class InputBench extends InputBenchmark {
  @Override
  protected Variant variant() {
    return Variant.TENHOU_3P;
  }
}
