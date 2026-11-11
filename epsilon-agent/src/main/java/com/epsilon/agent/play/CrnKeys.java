package com.epsilon.agent.play;

import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.Arrays;

/**
 * 判断のキーと、指数競争による行動の抽選。同じキーなら同じ手を選ぶ。
 *
 * <p>キーは (局のキー, 席, 判断のクラス, 席と判断のクラスごとの序数) で決め、判断 ID は使わない。判断 ID は応答できる席の数によって
 * 枝ごとにずれるので、共通乱数にならない。手 a ごとにキーと行動の番号から一様値 U(key, 行動の番号(a)) を作り、−log U / σ(a) が最小の手を選ぶ（σ(a) は手の確率）。これは Gumbel-max と同じ分布になる。同じ
 * キーと行動の番号には、主経路・写し・分岐で同じ一様値が付く。そのため、同じ局面では同じ手を選ぶ（候補の間の共通乱数）。
 */
public final class CrnKeys {
  /** 判断のクラスの数。手番系（手番・鳴いた後の打牌）と応答系（打牌への応答・槍槓・抜いた北への応答）の 2 つ。 */
  public static final int KIND_CLASSES = 2;

  private static final long ACTION_GAMMA = 0x9E3779B97F4A7C15L;

  private static final double UNIT = 0x1.0p-53;

  private CrnKeys() {}

  public static int kindClass(EngineDecisionKind kind) {
    return switch (kind) {
      case TURN, POST_CALL_DAHAI -> 0;
      case RESPONSE, CHANKAN, NUKI_RESPONSE -> 1;
    };
  }

  public static long decisionKey(long handKey, int seat, int kindClass, int ordinal) {
    long coordinate = ((long) ordinal << 3) | ((long) seat << 1) | kindClass;
    return SeedMixer.mix64(handKey ^ SeedMixer.mix64(coordinate));
  }

  /** 行動ごとの一様値 U(key, 行動の番号) ∈ (0, 1)。 */
  public static double uniform(long key, int code) {
    return ((SeedMixer.mix64(key + (code + 1L) * ACTION_GAMMA) >>> 11) + 0.5) * UNIT;
  }

  /**
   * 手の確率に従って行動の位置を選ぶ。確率が 0 の手は選ばない。
   *
   * @param legal 合法手の行動の番号
   * @param sigma 合法手の順に並んだ確率（{@code offset} の位置から読む）。和が 1 からずれていても、選ぶ手の分布は変わらない
   */
  public static int select(long key, int[] legal, float[] sigma, int offset) {
    int n = legal.length;
    if (n == 1) return 0;
    int best = -1;
    double bestRace = Double.POSITIVE_INFINITY;
    for (int slot = 0; slot < n; slot++) {
      float probability = sigma[offset + slot];
      if (!(probability > 0)) continue;
      double race = -Math.log(uniform(key, legal[slot])) / probability;
      if (race < bestRace) {
        bestRace = race;
        best = slot;
      }
    }
    if (best < 0) throw new IllegalStateException("Policy has no positive probability");
    return best;
  }

  /** 席と判断のクラスごとの序数。主経路・写し・分岐で同じだけ進め、枝を分けるときにコピーする。 */
  public static final class Cursor {
    private final int kindClasses;
    private final int[] ordinals;

    /**
     * 序数を 0 で始める。
     *
     * @param seats 席の数
     * @param kindClasses 判断のクラスの数（{@link CrnScheme#kindClasses}）
     */
    public Cursor(int seats, int kindClasses) {
      this(kindClasses, new int[seats * kindClasses]);
    }

    private Cursor(int kindClasses, int[] ordinals) {
      this.kindClasses = kindClasses;
      this.ordinals = ordinals;
    }

    public int next(int seat, int kindClass) {
      return ordinals[seat * kindClasses + kindClass]++;
    }

    public void reset() {
      Arrays.fill(ordinals, 0);
    }

    public Cursor copy() {
      return new Cursor(kindClasses, ordinals.clone());
    }
  }
}
