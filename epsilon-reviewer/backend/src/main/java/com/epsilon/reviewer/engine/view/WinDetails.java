package com.epsilon.reviewer.engine.view;

import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.List;

/**
 * 和了の採点の内訳。エンジンが局末に採点した結果から作る。
 *
 * @param han 役・ドラ・赤ドラ・裏ドラの翻の合計。役満なら 0
 * @param fu 符。役満なら 0
 * @param points 本場・供託・包を含まない和了点（ツモなら、払う席の支払いの合計）
 * @param yakuman 役満の倍率。役満でなければ 0
 * @param yaku 役とドラ。役満なら役満の役だけ
 * @param uraIndicators 局末にめくった裏ドラ表示牌の記号（表示牌の順）。立直での和了でなければ空
 */
public record WinDetails(
    int han, int fu, int points, int yakuman, List<Yaku> yaku, List<String> uraIndicators) {
  private static final ScoringYaku[] YAKU = ScoringYaku.values();
  private static final String[] WINDS = {"E", "S", "W", "N"};

  public WinDetails {
    yaku = List.copyOf(yaku);
    uraIndicators = List.copyOf(uraIndicators);
  }

  /**
   * 内訳に並べる役 1 つ。
   *
   * @param code 役のコード。{@link ScoringYaku} の名前で、自風と場風の役牌には風を付ける（YAKUHAI_SEAT_E など）。ドラは DORA・AKADORA・URADORA
   * @param han 翻。役満の役なら 0
   * @param yakuman 役満の倍率。役満でない役は 0
   */
  public record Yaku(String code, int han, int yakuman) {}

  /**
   * 局末の和了 1 つの内訳を作る。役は {@link ScoringYaku} の順に並べ、役満でなければ後ろにドラ・赤ドラ・裏ドラを足す。
   *
   * @param variant ルール（四人打ちか三人打ちか）
   * @param round 局の通し番号
   * @param outcome 局の結果
   * @param index 何番目の和了か（局の結果に並ぶ和了者の順）
   * @param score その和了の採点
   * @param uraIndicators 裏ドラ表示牌の記号
   */
  public static WinDetails of(
      Variant variant,
      int round,
      RoundOutcome outcome,
      int index,
      WinScore score,
      List<String> uraIndicators) {
    int winner = outcome.winner(index);
    List<Yaku> yaku = new ArrayList<>();
    for (ScoringYaku each : YAKU)
      if (score.has(each))
        yaku.add(
            new Yaku(
                code(each, variant, round, winner),
                each.han(score.closed()),
                each.yakuman() ? 1 : 0));
    if (score.yakuman() == 0) {
      addDora(yaku, "DORA", score.dora());
      addDora(yaku, "AKADORA", score.aka());
      addDora(yaku, "URADORA", score.ura());
    }
    return new WinDetails(
        score.han(),
        score.fu(),
        points(variant, round, outcome, index, score),
        score.yakuman(),
        yaku,
        uraIndicators);
  }

  /** 本場・供託・包を除いた和了点を求める。 */
  private static int points(
      Variant variant, int round, RoundOutcome outcome, int index, WinScore score) {
    int winner = outcome.winner(index);
    int dealer = variant.dealer(round);
    int[] delta = new int[variant.seats()];
    if (outcome.kind() == EndKind.TSUMO)
      Settlement.tsumo(variant, score.base(), winner, dealer, 0, RoundOutcome.NO_SEAT, delta);
    else
      Settlement.ron(
          variant, score.base(), winner, outcome.loser(), dealer, 0, RoundOutcome.NO_SEAT, delta);
    return delta[winner];
  }

  private static String code(ScoringYaku yaku, Variant variant, int round, int winner) {
    return switch (yaku) {
      case YAKUHAI_SEAT -> "YAKUHAI_SEAT_" + WINDS[variant.seatWind(round, winner) - Tiles.TON];
      case YAKUHAI_ROUND -> "YAKUHAI_ROUND_" + WINDS[variant.roundWind(round) - Tiles.TON];
      default -> yaku.name();
    };
  }

  private static void addDora(List<Yaku> yaku, String code, int count) {
    if (count > 0) yaku.add(new Yaku(code, count, 0));
  }
}
