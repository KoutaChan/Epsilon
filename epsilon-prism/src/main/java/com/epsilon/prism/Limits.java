package com.epsilon.prism;

import com.epsilon.agent.facts.WaitScores;
import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.Tiles;

/**
 * PRISM の形式が使う上限。ルールで決まる値はルール（{@link Variant}）とエンジン（epsilon-mahjong・epsilon-agent）から求め、ここで
 * 書き写さない。席の数や使う牌で変わる値はルールを受け取るメソッドにし、バッチの大きさを決める定数は全てのルールでの最大にする。範囲の外の
 * 値は書くときに例外にし、切り捨てない。
 *
 * <p>ここにはルールで決まる値だけを置く。表し方の選択（入力のイベントのトークンの容量、本場や翻をまとめ始める値、O の本場の上限）は、
 * それぞれの形式が持つ。
 */
public final class Limits {

  /** ドラ表示の最大（初めの表示 1 ＋ 槓ドラ 4）。 */
  public static final int MAX_INDICATORS = WinContext.MAX_INDICATORS;

  /**
   * 入力のイベントのトークンの最大（全てのルールでの最大。四人打ちでは O のイベント 106 ＋ 槓ドラの表示 4 = 110）。O のイベントは
   * エンジンの公開のイベントと同じで、槓ドラの表示を含まない。トークンはそれに槓ドラの表示を足したもので、初めのドラ表示は含まない。
   */
  public static final int MAX_EVENT_TOKENS = maxPublicEvents() + MAX_INDICATORS - 1;

  /** 行ごとに必ずあるトークン（局 1 つ、席ごと、牌の種類ごと）の数の最大（全てのルールでの最大。四人打ちでは 1 ＋ 4 ＋ 34 = 39）。 */
  public static final int MAX_FIXED_TOKENS = maxFixedTokens();

  /** 1 行のトークンの数の最大（全てのルールでの最大）。 */
  public static final int MAX_ROW_TOKENS = MAX_FIXED_TOKENS + MAX_EVENT_TOKENS;

  /** 合法手の最大（エンジンが席ごとの判断に持てる数）。これを超える判断は、取り込むときに例外にする。 */
  public static final int MAX_LEGAL_ACTIONS = Decisions.MAX_LEGAL;

  /** 1 つの判断が持つ遷移の最大（O の表し方での上限）。 */
  public static final int MAX_TRANSITIONS = ObservationSchema.MAX_TRANSITIONS;

  /** 和了 1 回で和了者が受け取る点（本場・供託を除く）の最大。親が全ての役満を重ねて和了したときの点にする。 */
  public static final int MAX_WIN_POINTS = ScoreMath.ron(ScoreMath.base(0, 0, ObservationSchema.MAX_YAKUMAN), true);

  /**
   * 1 局で放銃者が払う点（本場を除く）の最大。三家和了は途中流局なので、1 つの打牌をロンできるのは 2 人までで、そのうち親は 1 人だけ。
   * 親と子が最大の和了でロンしたときの点にする。
   */
  public static final int MAX_DEAL_IN_POINTS =
      MAX_WIN_POINTS + ScoreMath.ron(ScoreMath.base(0, 0, ObservationSchema.MAX_YAKUMAN), false);

  /** 形ごとの向聴数のうち最小のものの上限。門前なら七対子で 6 以下、副露があれば通常形で 8 − 2 = 6 以下になる。 */
  public static final int MAX_SHANTEN = 6;

  public static final int MAX_STANDARD_SHANTEN = 8;
  public static final int MAX_CHIITOI_SHANTEN = 6;
  public static final int MAX_KOKUSHI_SHANTEN = 13;

  /** 待ちの種類の最大（国士の 13 面）。 */
  public static final int MAX_WAIT_KINDS = WaitScores.MAX_WAITS;

  /** 見えている役牌の刻子・槓子の翻の合計の最大（ダブ風 2 ＋ 三元牌 3）。 */
  public static final int MAX_OPEN_YAKUHAI_HAN = 5;

  /** 役満でない役の翻の和（役ごとに、門前と副露の大きい方を足す）。 */
  private static final int MAX_YAKU_HAN = yakuHan();

  private Limits() {}

  /** 赤 5 の枚数。 */
  public static int akaTiles(Variant variant) {
    return Long.bitCount(variant.tiles().redKinds());
  }

  /** 供託の本数の最大（全員の持ち点を全て供託にしたときの本数）。 */
  public static int maxKyotaku(Variant variant) {
    return variant.seats() * variant.startPoints() / Variant.DEPOSIT;
  }

  /** 手牌と副露にあるドラの最大（表示 5 枚 × 同じ牌 4 枚 ＋ 赤 5 ＋ 抜いた北）。 */
  public static int maxDora(Variant variant) {
    int nuki = variant.has(RuleFlag.NUKI) ? Tiles.COPIES : 0;
    return MAX_INDICATORS * Tiles.COPIES + akaTiles(variant) + nuki;
  }

  /** 和了の翻（裏ドラを除く）の最大。役満でない役の翻の和にドラの最大を足す。 */
  public static int maxWinHan(Variant variant) {
    return MAX_YAKU_HAN + maxDora(variant);
  }

  private static int maxPublicEvents() {
    int events = 0;
    for (Variant variant : Variant.values()) {
      events = Math.max(events, variant.maxPublicEvents());
    }
    return events;
  }

  private static int maxFixedTokens() {
    int tokens = 0;
    for (Variant variant : Variant.values()) {
      tokens = Math.max(tokens, 1 + variant.seats() + Long.bitCount(variant.tiles().kinds()));
    }
    return tokens;
  }

  private static int yakuHan() {
    int han = 0;
    for (ScoringYaku yaku : ScoringYaku.values()) {
      if (!yaku.yakuman()) {
        han += Math.max(yaku.han(true), yaku.han(false));
      }
    }
    return han;
  }
}
