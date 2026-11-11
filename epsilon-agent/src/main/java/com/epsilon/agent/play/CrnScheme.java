package com.epsilon.agent.play;

import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.util.SeedMixer;

/**
 * 判断の共通乱数の使い方（{@link CrnKeys}）。{@link #SINGLE_RACE} は、行動方策のない卓（評価）での使い方。{@link #TWO_STAGE_FLOOR} は、
 * 兄弟の分岐（同じ局面から手だけを変えた続き）が同じ乱数を引く所を増やして、分岐の差のノイズを減らす使い方。
 *
 * <ul>
 *   <li>判断のクラス: SINGLE_RACE は手番系（手番・鳴いた後の打牌）と応答系の 2 つ。TWO_STAGE_FLOOR は鳴いた後の打牌を分けた 3 つで、
 *       鳴いた枝と鳴かなかった枝で手番の序数がずれない。
 *   <li>一様分布の混ぜ方: SINGLE_RACE は、打ち手の分布 σ に一様分布を割合 ε だけ混ぜた b = (1 − ε)·σ + ε/n（n は合法手の数）の上で 1 回の指数競争をする（{@link CrnKeys#select}）。TWO_STAGE_FLOOR は
 *       2 段の乱数で引く（{@link #select}）。どちらも分布は b になる。TWO_STAGE_FLOOR では一様分布に切り替えるかが σ と合法手に依らない
 *       ので、兄弟の分岐は同じキーの判断で一緒に切り替わる。
 * </ul>
 */
public enum CrnScheme {
  /** 判断のクラスは 2 つ。b の上で 1 回の指数競争をする（キーの式は {@link CrnKeys#decisionKey}）。 */
  SINGLE_RACE(CrnKeys.KIND_CLASSES),
  /** 判断のクラスは 3 つ。2 段の乱数で引く。 */
  TWO_STAGE_FLOOR(3);

  /** 2 段の引き方で、一様分布に切り替えるかを決める乱数に使う行動の番号。合法手の番号は 0 以上なので重ならない。 */
  private static final int FLOOR_COIN = -1;

  private final int kindClasses;

  CrnScheme(int kindClasses) {
    this.kindClasses = kindClasses;
  }

  /** 判断のクラスの数（{@link CrnKeys.Cursor} の大きさ）。 */
  public int kindClasses() {
    return kindClasses;
  }

  /** 判断の種類が属するクラスの番号。 */
  public int kindClass(EngineDecisionKind kind) {
    return switch (this) {
      case SINGLE_RACE -> CrnKeys.kindClass(kind);
      case TWO_STAGE_FLOOR ->
          switch (kind) {
            case TURN -> 0;
            case RESPONSE, CHANKAN, NUKI_RESPONSE -> 1;
            case POST_CALL_DAHAI -> 2;
          };
    };
  }

  /** 判断のキー。(局のキー, 席, 判断のクラス, 席と判断のクラスごとの序数) から作る。 */
  public long decisionKey(long handKey, int seat, int kindClass, int ordinal) {
    return switch (this) {
      case SINGLE_RACE -> CrnKeys.decisionKey(handKey, seat, kindClass, ordinal);
      case TWO_STAGE_FLOOR -> {
        long coordinate = ((long) ordinal << 4) | ((long) seat << 2) | kindClass;
        yield SeedMixer.mix64(handKey ^ SeedMixer.mix64(coordinate));
      }
    };
  }

  /**
   * 一様分布を割合 ε だけ混ぜた分布 b = (1 − ε)·σ + ε/n から、行動の位置を引く。SINGLE_RACE は b の上で 1 回の指数競争をして引く。
   * TWO_STAGE_FLOOR は 2 段で引く。まず、キーから作る乱数 U(key, −1) が ε 未満なら、合法手の中から一様に引く（U(key, 行動の番号) が
   * 最も大きい手）。そうでなければ σ の上で指数競争をする。σ の和が 0 のときも一様に引く。切り替えを決める乱数は行動ごとの一様値と
   * 独立なので、分布はちょうど b になる。
   *
   * @param key 判断のキー
   * @param legal 合法手の行動の番号
   * @param sigma 合法手の順の σ（和が 0 なら b は一様）
   * @param b σ に一様分布を混ぜた分布（{@link Behaviour#policy}。SINGLE_RACE はこれで引く）
   * @param epsilon 一様分布を混ぜる割合 ε
   * @return 選んだ位置
   */
  public int select(long key, int[] legal, float[] sigma, float[] b, double epsilon) {
    if (this == SINGLE_RACE) return CrnKeys.select(key, legal, b, 0);
    int n = legal.length;
    if (n == 1) return 0;
    boolean fired = CrnKeys.uniform(key, FLOOR_COIN) < epsilon;
    if (!fired && positive(sigma, n)) return CrnKeys.select(key, legal, sigma, 0);
    int best = 0;
    double largest = -1;
    for (int slot = 0; slot < n; slot++) {
      double u = CrnKeys.uniform(key, legal[slot]);
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
