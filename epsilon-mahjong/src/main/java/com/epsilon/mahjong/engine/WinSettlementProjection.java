package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 判断する席がいまツモ和了かロンをしたとして、裏ドラを数えない点数、点の動き、和了の後の順位を求める。1 席から見た卓
 * （{@link SeatView}）だけを読むので、エンジンの外（牌譜、1 席から見た卓、観測の記録を書く側）でも同じ値になる。使い回す可変の
 * オブジェクトで、求め直すたびに全てのフィールドを書き直す。同時に使えるのは 1 つのスレッドだけ。
 *
 * <p>点の動きは包を含み、場の供託は含まない。ツモの点の動きは本場を含めて 1 通りに決まる。ロンの点の動きは本場を除いて求める（ダブロン
 * では、本場を受け取る和了者が 1 人に決まらない）。順位は、精算の後の点に、和了者が受け取る場の供託を足して付ける。精算は、ツモなら
 * そのツモの精算、ロンなら和了者が 1 人で本場も受け取るとした精算を使う。
 */
public final class WinSettlementProjection {

  /** 和了の後の順位を付けるときに仮定した精算。名前と並びの順は PRISM の語彙が使うので変えない。 */
  public enum SettlementAssumption {
    /** ツモ。本場と場の供託を含めて精算が 1 つに決まる。 */
    EXACT_TSUMO,
    /** ロンの和了者が 1 人だけで、本場と場の供託を全て受け取る。 */
    SOLE_RON
  }

  private final Variant variant;
  private final WinScore score = new WinScore();

  /** [絶対席] → 点の動き。 */
  private final int[] payments;

  /** [絶対席] → 順位を付けるのに使った、精算の後の点。 */
  private final int[] after;

  private final int[] ranks;
  private SettlementAssumption assumption = SettlementAssumption.EXACT_TSUMO;
  private int pao = RoundOutcome.NO_SEAT;
  private boolean uraEligible;
  private int uraIndicators;

  public WinSettlementProjection(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    payments = new int[variant.seats()];
    after = new int[variant.seats()];
    ranks = new int[variant.seats()];
  }

  /**
   * 見る席が和了したとして、点数・点の動き・順位を求める。ツモは直前に自分がツモった牌で、ロンは直前の打牌・加槓の牌・抜いた北で
   * 和了する。打牌へのロンは山が尽きていれば河底、加槓へのロンは槍槓になる。合法かは確かめない。役が無いとき、ツモで直前が自分の
   * ツモでないとき、ロンで判断の時点が他家の応答でないときは IllegalStateException を投げる。卓か作業領域のルールがこのオブジェクトと
   * 違うときと、手のコードがツモ和了でもロンでもないときは IllegalArgumentException を投げる。
   *
   * @param view 和了する席から見た卓
   * @param winCode ツモ和了かロンの手のコード
   * @param ws ルールの計算に使う作業領域
   */
  public void load(SeatView view, int winCode, RuleWorkspace ws) {
    if (view.variant() != variant || ws.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + " but found " + view.variant() + " and " + ws.variant());
    }
    Action win = Action.of(winCode);
    boolean tsumo =
        switch (win.type()) {
          case TSUMO_AGARI -> true;
          case RON_AGARI -> false;
          default -> throw new IllegalArgumentException("Not a win: " + win);
        };
    int seat = view.seat();
    TurnEvent last = view.last();
    WinContext context = ws.context();
    loadHand(view, context);
    long reds = Legal.redFives(view);
    if (tsumo) {
      if (last.kind() != TurnEvent.Kind.DRAW || last.seat() != seat) {
        throw new IllegalStateException("Seat " + seat + " can tsumo only its own draw, not " + last.kind());
      }
      WinLegality.setTsumo(context, last, last.tile(), view.liveLeft(), view.events());
    } else {
      context.setWin(last.tile(), ronSource(view));
      if (last.red()) {
        reds |= Tiles.bit(last.tile());
      }
    }
    context.setRedFives(reds);
    if (!ws.scorer().score(context, score)) {
      throw new IllegalStateException(win + " has no yaku: " + context);
    }
    pao = WinLegality.pao(view, score, tsumo ? RoundOutcome.NO_SEAT : last.seat());
    int dealer = view.dealer();
    Arrays.fill(payments, 0);
    if (tsumo) {
      Settlement.tsumo(variant, score.base(), seat, dealer, view.honba(), pao, payments);
      System.arraycopy(payments, 0, after, 0, after.length);
      assumption = SettlementAssumption.EXACT_TSUMO;
    } else {
      Settlement.ron(variant, score.base(), seat, last.seat(), dealer, 0, pao, payments);
      Arrays.fill(after, 0);
      Settlement.ron(variant, score.base(), seat, last.seat(), dealer, view.honba(), pao, after);
      assumption = SettlementAssumption.SOLE_RON;
    }
    for (int other = 0; other < after.length; other++) {
      after[other] += view.score(other);
    }
    after[seat] += Variant.DEPOSIT * view.kyotaku();
    Placement.ranks(after, ranks);
    uraEligible = view.riichi(seat) != RiichiState.NONE;
    uraIndicators = view.indicatorCount();
  }

  /** 裏ドラを数えない点数。翻は役・表ドラ・赤・抜いた北から数える。 */
  public WinScore score() {
    return score;
  }

  public SettlementAssumption assumption() {
    return assumption;
  }

  /** 包の席。包が無いか、ロンで包になる席が放銃したなら {@code RoundOutcome.NO_SEAT}。 */
  public int pao() {
    return pao;
  }

  /** 立直しているなら true。このとき和了では裏ドラを数える。 */
  public boolean uraEligible() {
    return uraEligible;
  }

  /** 和了でめくる裏ドラ表示牌の数（いまめくってある表示牌の数と同じ）。 */
  public int uraIndicators() {
    return uraIndicators;
  }

  /** その絶対席の点の動き（受け取りが正）。 */
  public int payment(int seat) {
    return payments[seat];
  }

  /** その絶対席の、和了の後の順位（0 始まり。同点は起家に近い席が上）。 */
  public int rank(int seat) {
    return ranks[seat];
  }

  /** 和了の状況のうち、和了牌と赤以外の全て（純手牌・面子・抜いた北・風・立直・めくってある表示牌）を書く。 */
  private static void loadHand(SeatView view, WinContext context) {
    Variant variant = view.variant();
    int seat = view.seat();
    int round = view.round();
    context.clear();
    HandShape hand = context.hand();
    for (long kinds = variant.tiles().kinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      for (int copy = view.count(kind); copy > 0; copy--) {
        hand.add(kind);
      }
    }
    for (int index = 0; index < view.meldCount(seat); index++) {
      hand.addMeld();
      context.addMeld(Meld.of(view.meld(seat, index)));
    }
    context.setNuki(view.nuki(seat));
    context.setWinds(variant.seatWind(round, seat), variant.roundWind(round));
    context.setRiichi(view.riichi(seat), view.ippatsu(seat));
    for (int index = 0; index < view.indicatorCount(); index++) {
      context.addIndicator(Tiles.kind(view.indicator(index)));
    }
  }

  private static Source ronSource(SeatView view) {
    return switch (view.kind()) {
      case RESPONSE -> view.liveLeft() == 0 ? Source.HOUTEI : Source.RON;
      case CHANKAN -> Source.CHANKAN;
      case NUKI_RESPONSE -> Source.RON;
      case TURN, POST_CALL_DAHAI -> throw new IllegalStateException("No ron at a " + view.kind() + " decision");
    };
  }
}
