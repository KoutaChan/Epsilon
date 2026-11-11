package com.epsilon.prism.grp;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;

/**
 * 終盤の数え上げ（{@link PhiMode#LATE}）の設定。次の局がオーラス（南場の最後の局）かそれより後になる局末だけを、GRP の項目の局の結果の
 * ネットワークで数え上げ、ルールで精算する。
 *
 * @param from LATE のモードを使い始める反復（0 なら GRP のモードで始める）
 * @param cutoff LATE のモードの数え上げで、確率の積がこれより小さい組を足さない打ち切りの値
 * @param memo 数え上げた卓の Φ を覚えるメモのスロットの数。値には効かない。1 スロットは四人打ちで 40 バイト、三人打ちで 32 バイトなので、
 *     既定の 1,048,576 スロットでは 40 MiB と 32 MiB になる
 */
public record LateSettings(
    @Key(doc = "LATE のモードを使い始める反復（0 なら GRP のモードで始める）", value = "0", change = Change.FIXED) @Range(min = 0)
        int from,
    @Key(doc = "LATE のモードの数え上げで、確率の積がこれより小さい組を足さない打ち切りの値", value = "1e-5", change = Change.FIXED) @Range(min = 0, max = 0.01)
        double cutoff,
    @Key(doc = "数え上げた卓の Φ を覚えるメモのスロットの数", value = "1048576", change = Change.MACHINE)
        @Range(min = 1, max = MAX_MEMO) int memo) {

  /** メモのスロットの数の上限（2^26）。この上限なら、メモの Φ の配列の長さ（スロット数 × 席数）が int に収まる。 */
  public static final int MAX_MEMO = 1 << 26;

  public LateSettings {
    if (Integer.bitCount(memo) != 1) {
      throw new InvalidSetting("memo", "the late memo needs a power of two slots, got " + memo);
    }
  }
}
