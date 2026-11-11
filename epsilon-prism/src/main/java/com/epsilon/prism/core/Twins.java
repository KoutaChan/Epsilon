package com.epsilon.prism.core;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import java.util.Arrays;

/**
 * 同値な打牌の判定。種類（打牌か立直の打牌か）・牌種・赤かどうかが同じ打牌は、手出しかツモ切りかが違っても 1 つの候補として扱い、
 * 候補の集合でも逸脱でも同じ手と数える。打牌でない手は、それ自身とだけ同じ。手は行動の番号で表し、判定には {@code Observation.twins} を使う。
 * 同値な打牌のどちらを打つかは {@link TwinDiscard} で決める。
 */
public final class Twins {
  private Twins() {}

  /** 手番の打牌（立直を宣言する打牌を含む）なら true。 */
  public static boolean discard(Action action) {
    Action.Type type = action.type();
    return type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI;
  }

  /** 2 つの行動の番号が同じ手か、同値な打牌なら true。 */
  public static boolean same(int x, int y) {
    return x == y || Observation.twins(Action.of(x), Action.of(y));
  }

  /** 合法手の {@code a} 番目と {@code b} 番目が同じ手か、同値な打牌なら true。 */
  public static boolean same(int[] legal, int a, int b) {
    return a == b || same(legal[a], legal[b]);
  }

  /** 同値な打牌をまとめても、候補が 2 つ以上残るなら true。 */
  public static boolean branchable(int[] legal) {
    for (int a = 1; a < legal.length; a++) if (!same(legal, 0, a)) return true;
    return false;
  }

  /** 合法手の {@code a} 番目と同じ手か同値な打牌のうち、番号の最も小さい手の位置。 */
  public static int lowest(int[] legal, int a) {
    for (int b = 0; b < a; b++) if (same(legal, b, a)) return b;
    return a;
  }

  /**
   * 合法手の位置をそれぞれ {@link #lowest(int[], int)} に替え、昇順に並べる。
   *
   * @param legal 合法手の行動の番号
   * @param positions 合法手の位置（同値な打牌の組ごとに 1 つまで）
   * @return 替えた位置の昇順の新しい配列
   */
  public static int[] lowest(int[] legal, int[] positions) {
    int[] lowest = new int[positions.length];
    for (int i = 0; i < positions.length; i++) lowest[i] = lowest(legal, positions[i]);
    Arrays.sort(lowest);
    return lowest;
  }
}
