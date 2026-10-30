package com.epsilon.prism.core;

import com.epsilon.core.Action;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.prism.collect.Behaviour;
import com.epsilon.util.SeedMixer;
import java.util.List;

/**
 * 判断の共通乱数の使い方（{@link CrnKeys}）。{@link #V1} は行動方策のない対局（評価）の使い方。{@link #V2} は系譜の収集の使い方で、兄弟の
 * 分岐（同じ局面から手だけ違えた続き）が同じ乱数を引く所を増やして、分岐の差の雑音を減らす。
 *
 * <ul>
 *   <li>種別類: V1 は手番系（手番・鳴いた後の打牌）と応答系の 2 つ。V2 は鳴いた後の打牌を分けた 3 つで、鳴いた枝と鳴かなかった枝で手番の
 *       序数がずれない。
 *   <li>床: V1 は b = (1 − ε)·σ + ε/n の上の 1 回の指数競争（{@link CrnKeys#select}）。V2 は 2 段の硬貨（{@link #select}）で、分布は同じ b
 *       のまま、発火が σ と合法手に依らないので兄弟の分岐が同じ鍵の判断で一緒に発火する。
 * </ul>
 */
public enum CrnScheme {
  /** 種別類 2 つ、床は b の上の 1 回の競争（鍵の式は {@link CrnKeys#decisionKey}）。 */
  V1(CrnKeys.KIND_CLASSES),
  /** 種別類 3 つ、床は 2 段の硬貨。 */
  V2(3);

  /** 2 段の床の発火の硬貨の行動 ID（合法手の ID は 0 以上なので重ならない）。 */
  private static final int FLOOR_COIN = -1;

  private final int kindClasses;

  CrnScheme(int kindClasses) {
    this.kindClasses = kindClasses;
  }

  /** 種別類の数（{@link CrnKeys.Cursor} の大きさ）。 */
  public int kindClasses() {
    return kindClasses;
  }

  /** 判断の種別類。 */
  public int kindClass(EngineDecisionKind kind) {
    return switch (this) {
      case V1 -> CrnKeys.kindClass(kind);
      case V2 ->
          switch (kind) {
            case TURN -> 0;
            case RESPONSE, CHANKAN -> 1;
            case POST_CALL_DAHAI -> 2;
          };
    };
  }

  /** 判断の鍵（局のキー, 席, 種別類, 席×種別類ごとの序数）。 */
  public long decisionKey(long handKey, int seat, int kindClass, int ordinal) {
    return switch (this) {
      case V1 -> CrnKeys.decisionKey(handKey, seat, kindClass, ordinal);
      case V2 -> {
        long coordinate = ((long) ordinal << 4) | ((long) seat << 2) | kindClass;
        yield SeedMixer.mix64(handKey ^ SeedMixer.mix64(coordinate));
      }
    };
  }

  /**
   * 床 ε を混ぜた分布 b = (1 − ε)·σ + ε/n から行動の位置を引く。V1 は b を作って 1 回の競争で引く。V2 は 2 段の硬貨: 鍵の硬貨
   * U(key, 発火) < ε なら合法手の上の一様の競争（U の最も大きい手）、でなければ σ の競争。発火の硬貨は行動の一様値と独立なので、分布はちょうど
   * b になる。
   *
   * @param key 判断の鍵
   * @param legal 合法手
   * @param sigma 合法手の順の σ（和が 0 なら b は一様）
   * @param b σ に床を混ぜた分布（{@link Behaviour#policy}。V1 はこれで引く）
   * @param epsilon 床 ε
   * @return 選んだ位置
   */
  public int select(long key, List<Action> legal, float[] sigma, float[] b, double epsilon) {
    if (this == V1) return CrnKeys.select(key, legal, b, 0);
    int n = legal.size();
    if (n == 1) return 0;
    boolean fired = CrnKeys.uniform(key, FLOOR_COIN) < epsilon;
    if (!fired && positive(sigma, n)) return CrnKeys.select(key, legal, sigma, 0);
    int best = 0;
    double largest = -1;
    for (int slot = 0; slot < n; slot++) {
      double u = CrnKeys.uniform(key, legal.get(slot).toIndex());
      if (u > largest) {
        largest = u;
        best = slot;
      }
    }
    return best;
  }

  /** σ の和が正か（{@link Behaviour#policy} と同じ判定）。 */
  private static boolean positive(float[] sigma, int n) {
    double sum = 0;
    for (int a = 0; a < n; a++) sum += sigma[a];
    return sum > 0;
  }
}
