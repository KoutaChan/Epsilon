package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.model.PrismNetwork;
import java.util.Objects;

/**
 * 方策の記憶の設定。打つためのネットワーク A を、学習するネットワーク θ の重み付き平均にする。バージョン v を公開するたびに
 * A_v = (1 − η_v)·A_{v−1} + η_v·θ_v とする（{@link PrismNetwork#mixIn}）。平均を始めるバージョンより前は平均せず A = θ とする。
 * 始めるバージョンでは η = 1 なので、A はそのバージョンの θ と同じになる。
 */
@Choice(key = "scheme", doc = "収集で打つネットワークを、学習するネットワークの重みの平均にするか")
public sealed interface MemorySettings {

  /** 収集で打つネットワーク。 */
  MemoryNetwork collectWith();

  /** 生徒の学習を始めるネットワーク。TEMPO のアンカーにも使う。 */
  MemoryNetwork learnFrom();

  /**
   * バージョンを公開するときに混ぜる割合を返す。
   *
   * @param version 公開するバージョン
   * @return 混ぜる割合
   */
  Step step(int version);

  /** 平均を始めるバージョン。 */
  default int from() {
    return 1;
  }

  /** 平均を取るなら true。 */
  default boolean averages() {
    return !(this instanceof None);
  }

  /** 記録に書く方式の名前（none・ema）。 */
  default String scheme() {
    return switch (this) {
      case None none -> "none";
      case Ema ema -> "ema";
    };
  }

  /**
   * 1 つのバージョンでの混ぜ方。
   *
   * @param eta A_version = (1 − eta)·A_{version−1} + eta·θ_version の eta（1 なら A = θ）
   */
  record Step(double eta) {}

  /** 平均しない（A = θ）。 */
  @Case("NONE")
  record None() implements MemorySettings {

    @Override
    public MemoryNetwork collectWith() {
      return MemoryNetwork.ITERATE;
    }

    @Override
    public MemoryNetwork learnFrom() {
      return MemoryNetwork.ITERATE;
    }

    @Override
    public Step step(int version) {
      return new Step(1);
    }
  }

  /**
   * 指数移動平均。始めるバージョンより後は η = 1 − 2^(−1/halfLife)。
   *
   * @param halfLife 半減期（バージョンの数）
   * @param from 平均を始めるバージョン
   * @param collectWith 収集で打つネットワーク
   * @param learnFrom 生徒の学習を始めるネットワーク
   */
  @Case("EMA")
  record Ema(
      @Key(doc = "半減期（バージョンの数）", value = "8.0") @Range(min = 0, minInclusive = false) double halfLife,
      @Key(doc = "平均を始めるバージョン", value = "16") @Range(min = 1) int from,
      @Key(doc = "収集で打つネットワーク", value = "ITERATE") MemoryNetwork collectWith,
      @Key(doc = "生徒の学習を始めるネットワーク", value = "ITERATE") MemoryNetwork learnFrom)
      implements MemorySettings {

    public Ema {
      Objects.requireNonNull(collectWith, "collectWith");
      Objects.requireNonNull(learnFrom, "learnFrom");
    }

    @Override
    public Step step(int version) {
      return version <= from ? new Step(1) : new Step(1 - Math.pow(2, -1 / halfLife));
    }
  }
}
