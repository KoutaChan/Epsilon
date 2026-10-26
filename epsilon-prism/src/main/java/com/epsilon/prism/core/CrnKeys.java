package com.epsilon.prism.core;

import com.epsilon.core.Action;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.util.SeedMixer;
import java.util.Arrays;
import java.util.List;

/**
 * 局単位の構造キーと指数競争による行動の抽選（同じ種なら同じ手）。
 *
 * <p>キーは (局のキー, 席, 種別類, 席×種別類ごとの序数) で決まり、判断 ID を使わない。判断 ID は応答できる席の数で枝ごとにずれるため、
 * 共通乱数にならない。選択は argmin_a −log U(key, actionId(a)) / σ(a) で、Gumbel-max と同じ分布になる。同じキーと行動 ID
 * には主経路・写し・分岐で同じ一様値が付くので、同じ局面では同じ手を選ぶ（候補の間の共通乱数）。
 */
public final class CrnKeys {
  /** 手番系（TURN・POST_CALL_DAHAI）と応答系（RESPONSE・CHANKAN）。 */
  public static final int KIND_CLASSES = 2;

  private static final long ACTION_GAMMA = 0x9E3779B97F4A7C15L;

  private CrnKeys() {}

  public static int kindClass(EngineDecisionKind kind) {
    return kind == EngineDecisionKind.RESPONSE || kind == EngineDecisionKind.CHANKAN ? 1 : 0;
  }

  /** 局のキー。主経路とその局の写し・分岐が共有する。末尾の座標 0 は鍵の式の一部で、変えると鍵が変わる。 */
  public static long handKey(long iterationSeed, long gameId, int handOrdinal) {
    return CrnSeeds.derive(iterationSeed, CrnSeeds.Domain.HAND_KEY, gameId, handOrdinal, 0);
  }

  public static long decisionKey(long handKey, int seat, int kindClass, int ordinal) {
    long coordinate = ((long) ordinal << 3) | ((long) seat << 1) | kindClass;
    return SeedMixer.mix64(handKey ^ SeedMixer.mix64(coordinate));
  }

  /** 行動ごとの一様値 U(key, actionId) ∈ (0, 1)。 */
  public static double uniform(long key, int actionId) {
    return CrnSeeds.openUnit(SeedMixer.mix64(key + (actionId + 1L) * ACTION_GAMMA));
  }

  /**
   * σ に従って行動の位置を選ぶ。σ(a)=0 の手は選ばない。
   *
   * @param sigma 合法手の順に並んだ確率。正規化の誤差は選択の分布に効かない
   */
  public static int select(long key, List<Action> legal, float[] sigma, int offset) {
    int n = legal.size();
    if (n == 1) return 0;
    int best = -1;
    double bestRace = Double.POSITIVE_INFINITY;
    for (int slot = 0; slot < n; slot++) {
      float probability = sigma[offset + slot];
      if (!(probability > 0)) continue;
      double race = -Math.log(uniform(key, legal.get(slot).toIndex())) / probability;
      if (race < bestRace) {
        bestRace = race;
        best = slot;
      }
    }
    if (best < 0) throw new IllegalStateException("Policy has no positive probability");
    return best;
  }

  /** 席×種別類ごとの序数。主経路・写し・分岐で同じだけ進め、fork のときに写す。 */
  public static final class Cursor {
    private final int[] ordinals;

    public Cursor() {
      ordinals = new int[4 * KIND_CLASSES];
    }

    private Cursor(int[] ordinals) {
      this.ordinals = ordinals;
    }

    public int next(int seat, int kindClass) {
      return ordinals[seat * KIND_CLASSES + kindClass]++;
    }

    public void reset() {
      Arrays.fill(ordinals, 0);
    }

    public Cursor copy() {
      return new Cursor(ordinals.clone());
    }
  }
}
