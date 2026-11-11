package com.epsilon.mahjong.score;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.shape.WinningDecomposition.Wait;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 天鳳のルールで符を数える。通常形は、副底・雀頭・面子・待ちの符にツモ符か門前ロンの符を足し、10 符単位に切り上げる。30 符に
 * 満たなければ 30 符にする（喰い平和の形）。平和のツモはツモ符を付けずに 20 符、七対子は 25 符。
 */
final class Fu {

  static final int CHIITOITSU = 25;

  /** 副底。 */
  private static final int BASE = 20;

  private static final int PINFU_TSUMO = 20;
  private static final int TSUMO = 2;
  private static final int CLOSED_RON = 10;

  /** 単騎・嵌張・辺張の待ちの符。 */
  private static final int NARROW_WAIT = 2;

  /** 役牌（三元牌・自風・場風）の雀頭の符。連風牌は自風と場風で 2 回数える。 */
  private static final int VALUE_PAIR = 2;

  /** 中張牌の明刻の符。么九牌、暗刻、槓子のときは、それぞれの倍率をこれに掛ける。 */
  private static final int OPEN_SIMPLE_TRIPLET = 2;

  private static final int TERMINAL_FACTOR = 2;
  private static final int CONCEALED_FACTOR = 2;
  private static final int KAN_FACTOR = 4;
  private static final int UNIT = 10;
  private static final int MIN = 30;

  private Fu() {}

  /**
   * 刻子か槓子 1 つの符。
   *
   * @param kind 牌の種類
   * @param concealed 暗刻（暗槓）か
   * @param kan 槓子か
   */
  static int triplet(int kind, boolean concealed, boolean kan) {
    int fu = OPEN_SIMPLE_TRIPLET;
    if (Tiles.contains(Tiles.TERMINALS_AND_HONORS, kind)) {
      fu *= TERMINAL_FACTOR;
    }
    if (concealed) {
      fu *= CONCEALED_FACTOR;
    }
    if (kan) {
      fu *= KAN_FACTOR;
    }
    return fu;
  }

  /** マスクで指した種類の刻子（槓子は除く）について、符を合計する。 */
  static int triplets(long kinds, boolean concealed) {
    int fu = 0;
    for (long rest = kinds; rest != 0; rest &= rest - 1) {
      fu += triplet(Long.numberOfTrailingZeros(rest), concealed, false);
    }
    return fu;
  }

  /** 副露か暗槓 1 つの符。チーは 0。 */
  static int meld(Meld meld) {
    Meld.Type type = meld.type();
    return type == Meld.Type.CHI ? 0 : triplet(meld.base(), type.concealed(), type.kan());
  }

  static int pair(int kind, int seatWind, int roundWind) {
    int fu = Tiles.Group.of(kind) == Tiles.Group.DRAGON ? VALUE_PAIR : 0;
    if (kind == seatWind) {
      fu += VALUE_PAIR;
    }
    if (kind == roundWind) {
      fu += VALUE_PAIR;
    }
    return fu;
  }

  static int wait(Wait wait) {
    return switch (wait) {
      case TANKI, KANCHAN, PENCHAN -> NARROW_WAIT;
      case SHANPON, RYANMEN -> 0;
    };
  }

  /**
   * 通常形の符。
   *
   * @param parts 雀頭・面子・待ちの符の和
   * @param tsumo ツモ和了か
   * @param closed 門前か
   * @param pinfu 平和か
   * @return 切り上げた後の符
   */
  static int total(int parts, boolean tsumo, boolean closed, boolean pinfu) {
    if (pinfu && tsumo) {
      return PINFU_TSUMO;
    }
    int raw = BASE + parts + (tsumo ? TSUMO : closed ? CLOSED_RON : 0);
    return Math.max(MIN, (raw + UNIT - 1) / UNIT * UNIT);
  }
}
