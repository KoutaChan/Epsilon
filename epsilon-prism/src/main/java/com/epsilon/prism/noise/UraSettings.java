package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;

/** 立直の和了で終わった局の、局末の値の作り方。 */
@Choice(key = "mode", doc = "実際の裏ドラで学ぶか、裏ドラの期待値で学ぶか")
public sealed interface UraSettings {

  /**
   * この局末で期待値を求めるなら true を返す。期待値で学ぶなら常に求め、実際の値で学ぶなら監査に選んだ局末だけで求める。
   *
   * @param coin 局末の結果に依らない一様な乱数（0 以上 1 未満）
   */
  boolean averages(double coin);

  /**
   * 実際にめくった裏ドラで精算した値で学ぶ。
   *
   * @param sample 監査のために期待値も求める局末の割合
   */
  @Case("REALIZED")
  record Realized(@Key(doc = "監査のために期待値も求める局末の割合", value = "0.05") @Range(min = 0, max = 1) double sample)
      implements UraSettings {

    @Override
    public boolean averages(double coin) {
      return coin < sample;
    }
  }

  /** 裏ドラの数の分布で平均した値で学ぶ。局末の事実（{@code HandEndFacts}）は、実際の値のまま。 */
  @Case("EXPECTED")
  record Expected() implements UraSettings {

    @Override
    public boolean averages(double coin) {
      return true;
    }
  }
}
