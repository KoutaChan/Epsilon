package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 天鳳のルールで、和了できるかとその点数、包の席、荒牌での聴牌、流し満貫を判定する。
 *
 * <p>和了には役が要る。ドラ・赤・裏ドラ・抜いた北は役に数えず、HandScorer は役の無い手に false を返す。ロンできるのは形の上の待ち
 * （{@link Hand#waits}）にある牌だけで、見逃しの振聴（同巡内・立直の後）のときと、待ちのどれかが自分の河にある振聴のときはロンできない。
 * 点数は 1 回の採点で求める。裏ドラは立直の和了のときだけ、めくってある表示牌の数だけ数える。牌山が null のとき（牌山を持たない、
 * 1 席から見た卓）は裏ドラを数えない。
 */
final class WinLegality {

  /** 和了の形の向聴数。 */
  private static final int COMPLETE = -1;

  private static final long DRAGONS = Tiles.range(Tiles.HAKU, Tiles.CHUN);
  private static final long WINDS = Tiles.range(Tiles.TON, Tiles.PEI);

  private WinLegality() {}

  /**
   * 直前にツモった席が和了できるなら、点数を書いて true を返す。手は 14 枚相当。嶺上牌と北抜きの補充のツモは嶺上開花になり、最後の
   * 牌でも海底は付かない。山の最後の牌は海底になる。鳴き・暗槓・加槓・北抜きの無い局で自分の最初のツモなら、天和・地和になる。
   *
   * @param wall 裏ドラ表示牌を読む牌山。立直の和了の点数には裏ドラを含める。null なら裏ドラを数えない
   * @param out 和了の点数を書く先
   * @return 和了できれば true
   */
  static boolean tsumo(PublicState state, Hand hand, Wall wall, RuleWorkspace ws, WinScore out) {
    if (ws.analyzer().shanten(hand.shape()) != COMPLETE) {
      return false;
    }
    TurnEvent draw = state.last();
    int seat = draw.seat();
    WinContext context = load(state, hand, seat, hand.redFives(), ws);
    setTsumo(context, draw, draw.tile(), state.liveLeft(), state.events());
    addUra(state, wall, seat, context);
    return ws.scorer().score(context, out);
  }

  /**
   * 直前にツモった席の手で、引いた牌が別の牌だったとしたときのツモ和了の点数を、裏ドラを数えずに書く。出どころ（嶺上・海底）と
   * 第一ツモかどうかは、実際のツモと同じにする。実際に引いた牌が赤でも、その赤は数えない。
   *
   * @param kind 引いたとする牌の種類
   * @param red 引いたとする牌が赤の 5 なら true
   * @param out 和了の点数を書く先
   * @return 和了の形で役があれば true
   */
  static boolean tsumoAs(PublicState state, Hand hand, int kind, boolean red, RuleWorkspace ws, WinScore out) {
    TurnEvent draw = state.last();
    long reds = draw.red() ? hand.redFives() & ~Tiles.bit(draw.tile()) : hand.redFives();
    if (red) {
      reds |= Tiles.bit(kind);
    }
    WinContext context = load(state, hand, draw.seat(), reds, ws);
    context.hand().remove(draw.tile());
    setTsumo(context, draw, kind, state.liveLeft(), state.events());
    return ws.scorer().score(context, out);
  }

  /**
   * 直前の自分のツモについて、和了牌と出どころを和了の状況に書く。出どころは、嶺上牌と北抜きの補充なら嶺上開花（最後の牌でも海底は
   * 付かない）、山の最後の牌なら海底、それ以外はツモ。鳴き・暗槓・加槓・北抜きの無い局で自分の最初のツモなら、天和・地和になる印も付ける。
   *
   * @param context 書く先の和了の状況
   * @param draw 直前のツモ
   * @param kind 和了牌の種類（引いた牌と違ってよい）
   * @param liveLeft ツモの後の山の残りの枚数
   * @param events 局の公開イベント
   */
  static void setTsumo(WinContext context, TurnEvent draw, int kind, int liveLeft, EventLog events) {
    Source source = draw.rinshan() ? Source.RINSHAN : liveLeft == 0 ? Source.HAITEI : Source.TSUMO;
    context.setWin(kind, source);
    context.setFirstDraw(
        source == Source.TSUMO && events.interruptions() == 0 && events.riverSize(draw.seat()) == 0);
  }

  /**
   * 直前の打牌・加槓の牌・抜いた北でロンできるなら、点数を書いて true を返す。
   *
   * @param seat ロンする席
   * @param kind ロンする牌の種類
   * @param red ロンする牌が赤の 5 なら true
   * @param source 和了の種類。ロン（抜いた北へのロンを含む）・河底・槍槓のどれか
   * @param wall 裏ドラ表示牌を読む牌山。立直の和了の点数には裏ドラを含める。null なら裏ドラを数えない
   * @param out 和了の点数を書く先
   * @return ロンできれば true
   */
  static boolean ron(
      PublicState state,
      Hand hand,
      int seat,
      int kind,
      boolean red,
      Source source,
      Wall wall,
      RuleWorkspace ws,
      WinScore out) {
    if (!claimable(state, hand, seat, kind)) {
      return false;
    }
    WinContext context = loadRon(state, hand, seat, kind, red, source, ws);
    addUra(state, wall, seat, context);
    return ws.scorer().score(context, out);
  }

  /** {@link #ron} と同じ判定をし、裏ドラを数えない点数を書く。 */
  static boolean ronWithoutUra(
      PublicState state, Hand hand, int seat, int kind, boolean red, Source source, RuleWorkspace ws, WinScore out) {
    return claimable(state, hand, seat, kind)
        && ws.scorer().score(loadRon(state, hand, seat, kind, red, source, ws), out);
  }

  /**
   * 和了の包の席を返す。面子を足した順に数え、大三元なら 3 つ目の三元牌の面子、大四喜なら 4 つ目の風牌の面子を鳴かせた席が包になる。
   * その面子が暗槓なら包は無く、加槓なら元のポンを鳴かせた席が包になる。包になる席がロンで放銃したときは包は無く、放銃者が全額を払う
   * 普通のロンになる（天鳳の牌譜にも包の席は無い）。
   *
   * @param winner 和了者の席から見た卓
   * @param score 和了の点数
   * @param loser ロンの放銃者。ツモなら {@code RoundOutcome.NO_SEAT}
   * @return 包の席。無ければ {@code RoundOutcome.NO_SEAT}
   */
  static int pao(SeatView winner, WinScore score, int loser) {
    int liable =
        score.has(ScoringYaku.DAISANGEN)
            ? liable(winner, DRAGONS)
            : score.has(ScoringYaku.DAISUUSHII) ? liable(winner, WINDS) : RoundOutcome.NO_SEAT;
    return liable == loser ? RoundOutcome.NO_SEAT : liable;
  }

  /**
   * 荒牌で聴牌なら true。純手牌で 4 枚持っている種類を待つ形は不聴だが、その種類は形の上の待ちに入らない（手に 4 枚ある種類は
   * 引けない）ので、待ちが残るかだけを見ればよい。副露で 3 枚使った種類の単騎は、純手牌ではその牌が 1 枚なので聴牌になる。
   */
  static boolean tenpai(Hand hand) {
    return hand.waits() != 0;
  }

  /** 流し満貫なら true。その席の河が空でなく、全てが么九牌で、どれも鳴かれていないときに限る（自分が鳴いていてもよい）。 */
  static boolean nagashi(EventLog events, int seat) {
    int size = events.riverSize(seat);
    for (int index = 0; index < size; index++) {
      int event = events.river(seat, index);
      if (events.calledBy(event) != EventLog.NONE
          || !Tiles.contains(Tiles.TERMINALS_AND_HONORS, events.tile(event))) {
        return false;
      }
    }
    return size > 0;
  }

  /** 形の上の待ちにあり、見逃しの振聴でも河の振聴でもなければ true。 */
  private static boolean claimable(PublicState state, Hand hand, int seat, int kind) {
    long waits = hand.waits();
    return Tiles.contains(waits, kind) && !hand.missedRon() && (waits & state.riverKinds(seat)) == 0;
  }

  private static WinContext loadRon(
      PublicState state, Hand hand, int seat, int kind, boolean red, Source source, RuleWorkspace ws) {
    long reds = red ? hand.redFives() | Tiles.bit(kind) : hand.redFives();
    WinContext context = load(state, hand, seat, reds, ws);
    context.setWin(kind, source);
    return context;
  }

  /**
   * 和了の状況のうち和了牌以外（手・面子・赤・抜いた北・風・立直・表示牌）を、作業領域が持つ和了の状況に書く。表示牌はめくってある
   * ものだけを書き、保留の槓ドラは数えない。
   *
   * @param redFives 和了の手（和了牌を含む）にある赤の 5 の種類のマスク
   * @param ws 書く先の和了の状況を持つ作業領域
   * @return 書いた和了の状況
   */
  private static WinContext load(PublicState state, Hand hand, int seat, long redFives, RuleWorkspace ws) {
    Variant variant = state.variant();
    WinContext context = ws.context();
    context.clear();
    context.hand().copyFrom(hand.shape());
    for (int index = 0; index < hand.meldCount(); index++) {
      context.addMeld(Meld.of(hand.meld(index)));
    }
    context.setRedFives(redFives);
    context.setNuki(hand.nuki());
    context.setWinds(variant.seatWind(state.round(), seat), variant.roundWind(state.round()));
    context.setRiichi(state.riichi(seat), state.ippatsu(seat));
    Dora dora = state.dora();
    for (int index = 0; index < dora.count(); index++) {
      context.addIndicator(Tiles.kind(dora.indicator(index)));
    }
    return context;
  }

  /** 立直の和了なら、めくってある表示牌のそれぞれに対応する裏ドラ表示牌を和了の状況に足す。 */
  private static void addUra(PublicState state, Wall wall, int seat, WinContext context) {
    if (wall == null || state.riichi(seat) == RiichiState.NONE) {
      return;
    }
    for (int index = 0; index < state.dora().count(); index++) {
      context.addUraIndicator(Tiles.kind(wall.ura(index)));
    }
  }

  /**
   * 三元牌か風牌の面子を足した順に数え、三元牌なら 3 つ目、風牌なら 4 つ目の面子を鳴かせた席を返す。その面子が暗槓のときと、
   * その数まで届かないときは、包は無い。
   */
  private static int liable(SeatView winner, long group) {
    int seat = winner.seat();
    int sets = 0;
    for (int index = 0; index < winner.meldCount(seat); index++) {
      Meld meld = Meld.of(winner.meld(seat, index));
      if (Tiles.contains(group, meld.base()) && ++sets == Long.bitCount(group)) {
        Meld.RelativeSource source = meld.source();
        return source == Meld.RelativeSource.NONE
            ? RoundOutcome.NO_SEAT
            : source.seat(seat, winner.variant().seats());
      }
    }
    return RoundOutcome.NO_SEAT;
  }
}
