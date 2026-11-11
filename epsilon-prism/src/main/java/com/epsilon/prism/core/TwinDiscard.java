package com.epsilon.prism.core;

import com.epsilon.mahjong.action.Action;
import java.util.Arrays;

/**
 * 同値な打牌（{@link Twins}）を手出しでもツモ切りでも切れるとき、どちらで切るか。ネットワークは同値な打牌に同じ値を付けるので、切る方は
 * この設定で決める。
 *
 * <p>ネットワークの入力は、この設定に依らない。同値な打牌の組は、組の中で番号の小さい手（手出し）の遷移 1 つを読む。その遷移の値は収集で
 * 切った方の続きから学ぶので、TSUMOGIRI で集めた系譜では、手出しと書いた遷移の値がツモ切りの続きの値になる。
 */
public enum TwinDiscard {
  /** ツモ切りで切る。 */
  TSUMOGIRI,
  /** 手出しで切る。 */
  TEDASHI;

  /**
   * 選んだ手を、同値な打牌のうち切る方の手に替える。打牌でない手と、切る方の手が合法手に無い打牌は替えない。種類（打牌か立直の打牌か）・
   * 牌・赤の選び方は変わらない。
   *
   * @param legal 合法手の行動の番号
   * @param chosen 選んだ手の、合法手の中の位置
   * @return 打つ手の、合法手の中の位置
   */
  public int play(int[] legal, int chosen) {
    Action action = Action.of(legal[chosen]);
    boolean tsumogiri = this == TSUMOGIRI;
    if (!Twins.discard(action) || action.selection().tsumogiri() == tsumogiri) return chosen;
    int twin =
        Action.discard(
            action.tile(), action.selection().red(), tsumogiri, action.type() == Action.Type.RIICHI_DAHAI);
    for (int a = 0; a < legal.length; a++) if (legal[a] == twin) return a;
    return chosen;
  }

  /**
   * 合法手の位置をそれぞれ切る方の手に替え（{@link #play(int[], int)}）、昇順に並べる。
   *
   * @param legal 合法手の行動の番号
   * @param positions 合法手の位置（同値な打牌の組ごとに 1 つまで）
   * @return 替えた位置の昇順の新しい配列
   */
  public int[] play(int[] legal, int[] positions) {
    int[] played = new int[positions.length];
    for (int i = 0; i < positions.length; i++) played[i] = play(legal, positions[i]);
    Arrays.sort(played);
    return played;
  }
}
