package com.epsilon.prism;

import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.calculate.scoring.YakuBits;
import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.RoundLimits;
import com.epsilon.core.Tile;
import com.epsilon.engine.ScorePayments;
import com.epsilon.engine.WinPointDeltas;

/**
 * 規則とエンジンから数えた上限。記録 O の欄の範囲と入力の語彙は、ここと {@link RoundLimits}（局の山・副露・槓・打牌・公開の出来事の上限。
 * エンジンと共有するので写しを持たない）から作る。範囲外の値は書くときに例外にし、切り捨てない。
 *
 * <p>規則で決まる事実だけを置く。表し方の選び（入力の出来事トークンの容量や本場・翻のまとめ始め、O の本場の上限）は、それぞれの schema
 * が持つ。
 */
public final class Limits {

  /** ドラ表示の最大（初めの表示 1 ＋ 槓ドラ 4）。 */
  public static final int MAX_INDICATORS = DoraState.MAX_INDICATORS;

  /** 赤牌の数（萬・筒・索の 5 に 1 枚ずつ）。 */
  public static final int AKA_TILES = 3;

  /**
   * 入力の出来事トークンの最大（O の出来事 106 ＋ 槓ドラの表示 4 = 110）。O の出来事はエンジンの公開の出来事（{@link
   * RoundLimits#MAX_PUBLIC_EVENTS}、槓ドラの表示を含まない）と同じで、トークンはそれに槓ドラの表示を足し、初めのドラ表示は含まない。
   */
  public static final int MAX_EVENT_TOKENS = RoundLimits.MAX_PUBLIC_EVENTS + MAX_INDICATORS - 1;

  /** 合法手の最大（エンジンの合法手の箱 EngineActionBuffer の容量と同じ）。超える判断は捕捉で例外にする。 */
  public static final int MAX_LEGAL_ACTIONS = 32;

  /** 1 判断の遷移の最大（EngineDecisionBuffer の容量と同じ）。 */
  public static final int MAX_TRANSITIONS = 192;

  /** チー・ポン 1 つの後の打牌の最大（13 枚からチー・ポンで 2 枚を出した残り 11 枚。打牌は手の牌ごとに 1 つ）。 */
  public static final int MAX_CALL_DISCARDS = 11;

  /** 供託の本数の最大（持ち点の総和 10 万点を全て供託にした数）。 */
  public static final int MAX_KYOTAKU = 100;

  /** 手と副露のドラの最大（表示 5 × 同じ牌 4 枚 ＋ 赤 3 枚）。 */
  public static final int MAX_DORA = MAX_INDICATORS * Tile.TILES_PER_TYPE + AKA_TILES;

  /** 和了の翻（裏なし）の最大（役満でない役の翻の和 ＋ ドラの最大）。 */
  public static final int MAX_WIN_HAN = sumOfYakuHan() + MAX_DORA;

  /** 符の符号の最大（110 符）。 */
  public static final int MAX_FU_CODE = ScoreMath.encodeFuCode(110);

  /** 役満の倍率の最大（全ての役満が重なったとき）。 */
  public static final int MAX_YAKUMAN = YakuBits.yakumanCount(-1L);

  /** 和了 1 回の点（本場・供託を除く和了者の受け取り）の最大（親の、全ての役満が重なった和了）。 */
  public static final int MAX_WIN_POINTS =
      ScorePayments.ronPoints(WinPointDeltas.MAX_BASE_POINTS, true);

  /**
   * 1 局で放銃に払う点（本場を除く）の最大。三家和了は途中流局なので、1 つの打牌をロンできるのは 2 人までで、親は 1 人（親と子が最大の
   * 和了でロンした）。
   */
  public static final int MAX_DEAL_IN_POINTS =
      MAX_WIN_POINTS + ScorePayments.ronPoints(WinPointDeltas.MAX_BASE_POINTS, false);

  /** 最小の向聴の最大（七対子の 6。副露があれば通常形は 8 − 2 = 6 以下）。 */
  public static final int MAX_SHANTEN = 6;

  public static final int MAX_STANDARD_SHANTEN = 8;
  public static final int MAX_CHIITOI_SHANTEN = 6;
  public static final int MAX_KOKUSHI_SHANTEN = 13;

  /** 待ちの種類の最大（国士の 13 面）。 */
  public static final int MAX_WAIT_KINDS = 13;

  /** 1 席の残りツモの最大（山 70 を 4 席で分けた切り上げ）。 */
  public static final int MAX_DRAWS_PER_SEAT =
      (RoundLimits.MAX_WALL + GameState.NUM_PLAYERS - 1) / GameState.NUM_PLAYERS;

  /** 公開の役牌の刻子・槓子の翻の最大（ダブ風 2 ＋ 三元牌 3）。 */
  public static final int MAX_OPEN_YAKUHAI_HAN = 5;

  /** 面子の正準 ID の数（種類 × 牌 × 元 × チーで鳴いた位置 3 × 赤の由来）。 */
  public static final int MELD_IDS =
      Meld.Type.values().length
          * Tile.NUM_TILE_TYPES
          * Meld.RelativeSource.values().length
          * 3
          * Meld.AkaSource.values().length;

  private Limits() {}

  private static int sumOfYakuHan() {
    int han = 0;
    for (ScoringYaku yaku : ScoringYaku.values()) {
      if (!yaku.isYakuman()) {
        han += Math.max(yaku.han(true), yaku.han(false));
      }
    }
    return han;
  }
}
