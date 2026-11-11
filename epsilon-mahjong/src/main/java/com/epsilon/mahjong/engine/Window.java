package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Objects;

/**
 * 他家の応答（打牌への応答、加槓への応答（槍槓）、抜いた北への応答）。直前のイベントへの応答を求めて、応答する席ごとの判断を作る。
 * 選ばれた手から優先の順に実際に行う手を決めて、応答を締める。暗槓には応答を求めない（国士無双でもロンできない）。
 *
 * <p>打牌への応答: 応答する席は、打った席を除いて絶対席の昇順に並ぶ。合法手の順は、ロン → ポン（手の赤を使う方が先）→ 大明槓 →
 * チー（上家の打牌だけ。順子の最小の種類の昇順で、それぞれ手の赤を使う方が先）→ 見送り。最後の打牌（山の残り 0）へのロンは河底で、
 * この打牌は鳴けない。立直中の席も鳴けない。面子にできない種類（北を鳴けないルールの北）はポン・大明槓できない。鳴いた後に打てる牌が
 * 残らないチーは合法手に入れない。ポンの後は純手牌が 13・10・7・4 枚なので、鳴いた種類のほかの牌が必ず残る。
 *
 * <p>加槓と抜いた北への応答: 応答する席は、宣言した席の次の席から手番の順に並び、合法手はロン → 見送り。加槓へのロンは槍槓になる。
 * 抜いた北へのロンは打牌へのロンと同じ形で、槍槓は付かず、宣言の前の一発は残る。
 *
 * <p>見送りは、ほかに選べる手があるときだけ合法手に入れる。合法手の無い席は判断を持たない。判断 ID は、判断を持つ席だけが順に使う。
 * 優先はロン ＞ ポン・大明槓 ＞ チーで、ロンを選んだ席は全員が和了する（三家和のあるルールで 3 人なら途中流局）。
 *
 * <p>ロンが無ければ応答を締める。打牌が立直の宣言牌なら、供託を払って立直を成立させる（宣言牌がその席の最初の打牌で、鳴き・暗槓・
 * 加槓・北抜きがまだ無ければダブル立直）。和了牌（形の上の待ち）を見送った席は同巡振聴にし、立直中なら局の終わりまで振聴にする。役が
 * 無いか振聴でロンできなかった席も、見送ったものとして数える。判断する席の無い応答も、同じように締める。
 */
final class Window {

  /** 締めた応答の結果。 */
  enum Outcome {
    /** 誰も応じなかった。 */
    PASS,
    /** チー・ポン・大明槓が通った。鳴いた席と鳴きの手のコードを読める。 */
    CALL,
    /** ロン（ダブロンを含む）。 */
    RON,
    /** 三家和の途中流局。 */
    TRIPLE_RON
  }

  private static final int RON_CODE = Action.code(Type.RON_AGARI);
  private static final int PASS_CODE = Action.code(Type.PASS);

  /** 判断を持たない席を表す値。 */
  private static final int NO_SLOT = -1;

  private final Variant variant;
  private final int seats;

  /** ロンの和了者。放銃者から手番が近い順に並ぶ。 */
  private final int[] winners;

  private int winnerCount;
  private Outcome outcome = Outcome.PASS;
  private int caller;
  private int call;

  Window(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    winners = new int[seats - 1];
  }

  /** ロンの和了者の数（ロンか三家和のとき 1 以上、ほかは 0）。 */
  int winnerCount() {
    return winnerCount;
  }

  /**
   * 和了者の席を返す。和了者は放銃者から手番が近い順に並び、本場と場の供託は最初の和了者が受け取る。
   *
   * @param i 何番目の和了者か（0 始まり）
   */
  int winner(int i) {
    return winners[Objects.checkIndex(i, winnerCount)];
  }

  /** 鳴いた席。鳴きが通っていなければ IllegalStateException を投げる。 */
  int caller() {
    requireCall();
    return caller;
  }

  /** 通った鳴きの手のコード。鳴きが通っていなければ IllegalStateException を投げる。 */
  int call() {
    requireCall();
    return call;
  }

  /**
   * 直前のイベント（打牌・加槓・北抜き）への応答を求め、応答する席ごとの判断を作る。ロンが合法なら、その点数を判断に持たせる。
   * ほかのイベントの後には応答が無いので、IllegalStateException を投げる。
   *
   * @param wall 裏ドラ表示牌を読む牌山。立直の和了の点数には裏ドラを含める。null なら裏ドラを数えない
   * @param id 次に使う判断 ID
   * @param calls false なら、打牌への応答にロンだけを出し、鳴きを出さない
   * @return 次に使う判断 ID（作った判断の数だけ進めたもの）
   */
  long open(
      PublicState state, Hand[] hands, Wall wall, Decisions decisions, long id, RuleWorkspace ws, boolean calls) {
    TurnEvent last = state.last();
    int actor = last.seat();
    decisions.clear();
    long next = id;
    boolean kakan = last.kind() == TurnEvent.Kind.KAN && last.kanKind() == TurnEvent.KanKind.KAKAN;
    if (last.kind() == TurnEvent.Kind.DISCARD) {
      boolean lastDiscard = state.liveLeft() == 0;
      Source source = ronSource(state);
      for (int seat = 0; seat < seats; seat++) {
        if (seat == actor) {
          continue;
        }
        decisions.open(seat, EngineDecisionKind.RESPONSE);
        int options = addRon(state, hands[seat], seat, last, source, wall, decisions, ws);
        if (calls && !lastDiscard && state.riichi(seat) == RiichiState.NONE) {
          options += addCalls(state, hands[seat], seat, last, decisions);
        }
        next = finish(decisions, options, next);
      }
    } else if (kakan || last.kind() == TurnEvent.Kind.NUKI) {
      EngineDecisionKind kind = kakan ? EngineDecisionKind.CHANKAN : EngineDecisionKind.NUKI_RESPONSE;
      Source source = ronSource(state);
      for (int offset = 1; offset < seats; offset++) {
        int seat = (actor + offset) % seats;
        decisions.open(seat, kind);
        next = finish(decisions, addRon(state, hands[seat], seat, last, source, wall, decisions, ws), next);
      }
    } else {
      throw new IllegalStateException("No response follows " + last.kind() + " " + last.kanKind());
    }
    return next;
  }

  /**
   * 直前のイベントへのロンが、どの種類の和了になるかを返す。打牌へのロンは、山の残りが 0 なら河底、それ以外はロン。加槓へのロンは
   * 槍槓、抜いた北へのロンはロン。
   *
   * @throws IllegalStateException 直前のイベントにロンできないとき
   */
  static Source ronSource(PublicState state) {
    TurnEvent last = state.last();
    return switch (last.kind()) {
      case DISCARD -> state.liveLeft() == 0 ? Source.HOUTEI : Source.RON;
      case KAN -> {
        if (last.kanKind() != TurnEvent.KanKind.KAKAN) {
          throw new IllegalStateException("No ron answers a " + last.kanKind());
        }
        yield Source.CHANKAN;
      }
      case NUKI -> Source.RON;
      case NONE, DRAW -> throw new IllegalStateException("No ron answers " + last.kind());
    };
  }

  /**
   * 他家の応答を締めて、実際に行う手を決める。ロン、ポン・大明槓、チーの順に優先し、ロンを選んだ席は全員が和了する。ロンが無ければ、
   * 宣言牌の立直を成立させて供託を払い、和了牌を見送った席に見逃しの印を付ける。
   *
   * @param choices 席ごとに選んだ手（合法手の何番目か）
   * @param resolution 席ごとに、選んだ手が行われたかを書く先
   * @return 締めた結果。誰も応じなかったとき（応答を求めた席が無いときを含む）は {@code PASS}
   */
  Outcome close(PublicState state, Hand[] hands, Decisions decisions, int[] choices, Resolution resolution) {
    TurnEvent last = state.last();
    int actor = last.seat();
    winnerCount = 0;
    int meldSlot = NO_SLOT;
    int chiSlot = NO_SLOT;
    for (int offset = 1; offset < seats; offset++) {
      int seat = (actor + offset) % seats;
      int slot = decisions.slotOf(seat);
      if (slot == NO_SLOT) {
        continue;
      }
      switch (chosen(decisions, choices, slot)) {
        case RON_AGARI -> winners[winnerCount++] = seat;
        case PON, DAIMINKAN -> meldSlot = meldSlot == NO_SLOT ? slot : meldSlot;
        case CHI -> chiSlot = slot;
        default -> {}
      }
    }
    int callSlot = meldSlot != NO_SLOT ? meldSlot : chiSlot;
    if (winnerCount > 0) {
      outcome =
          winnerCount == AbortiveDraw.TRIPLE_RON_WINNERS && variant.aborts(AbortiveDraw.TRIPLE_RON)
              ? Outcome.TRIPLE_RON
              : Outcome.RON;
    } else if (callSlot != NO_SLOT) {
      outcome = Outcome.CALL;
      caller = decisions.seat(callSlot);
      call = decisions.legal(callSlot, choices[callSlot]);
    } else {
      outcome = Outcome.PASS;
    }
    resolve(decisions, choices, resolution);
    if (outcome == Outcome.PASS || outcome == Outcome.CALL) {
      closeWithoutRon(state, hands);
    }
    return outcome;
  }

  /** 判断する席の無い応答を、誰も応じなかったとして締める。選んだ手が行われたかは書かない。 */
  void pass(PublicState state, Hand[] hands) {
    winnerCount = 0;
    outcome = Outcome.PASS;
    closeWithoutRon(state, hands);
  }

  void copyFrom(Window other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    System.arraycopy(other.winners, 0, winners, 0, winners.length);
    winnerCount = other.winnerCount;
    outcome = other.outcome;
    caller = other.caller;
    call = other.call;
  }

  /** ロンが合法なら合法手に足し、足した数を返す。 */
  private static int addRon(
      PublicState state,
      Hand hand,
      int seat,
      TurnEvent last,
      Source source,
      Wall wall,
      Decisions decisions,
      RuleWorkspace ws) {
    if (!WinLegality.ron(state, hand, seat, last.tile(), last.red(), source, wall, ws, ws.score())) {
      return 0;
    }
    decisions.addWin(RON_CODE, ws.score());
    return 1;
  }

  /** 打牌へのポン・大明槓・チーのうち合法なものを足し、足した数を返す。 */
  private int addCalls(PublicState state, Hand hand, int seat, TurnEvent discard, Decisions decisions) {
    int kind = discard.tile();
    int count = hand.count(kind);
    long reds = hand.redFives();
    boolean red = Tiles.contains(reds, kind);
    int added = 0;
    if (Tiles.contains(variant.meldableKinds(), kind)) {
      if (red && count >= 2) {
        decisions.add(Action.code(Type.PON, kind, TileSelection.AKA));
        added++;
      }
      if (count - (red ? 1 : 0) >= 2) {
        decisions.add(Action.code(Type.PON, kind, TileSelection.NON_AKA));
        added++;
      }
      if (count == Tiles.COPIES - 1 && state.kans() < Variant.MAX_KANS) {
        decisions.add(Action.code(Type.DAIMINKAN, kind, TileSelection.NOT_APPLICABLE));
        added++;
      }
    }
    Tiles.Group suit = Tiles.Group.of(kind);
    if (variant.has(RuleFlag.CHI) && seat == (discard.seat() + 1) % seats && suit.suited()) {
      int span = Meld.Type.CHI.tiles() - 1;
      for (int base = Math.max(suit.first(), kind - span); base <= Math.min(kind, suit.last() - span); base++) {
        added += addChi(hand, base, kind, reds, decisions);
      }
    }
    return added;
  }

  /**
   * チーが合法なら足し、足した数を返す。合法なのは、手に順子の残りの 2 枚があり、鳴いた後に打てる牌（喰い替えにならない牌）が残るとき。
   * 赤の 5 を使うチーと使わないチーは別の手として足す。
   *
   * @param base 順子の最小の種類
   * @param called 鳴く牌の種類
   * @param reds 純手牌にある赤の 5 の種類のマスク
   */
  private static int addChi(Hand hand, int base, int called, long reds, Decisions decisions) {
    long left = hand.shape().kinds();
    int redFive = TurnEvent.NONE;
    for (int kind = base; kind < base + Meld.Type.CHI.tiles(); kind++) {
      if (kind == called) {
        continue;
      }
      int count = hand.count(kind);
      if (count == 0) {
        return 0;
      }
      if (count == 1) {
        left &= ~Tiles.bit(kind);
      }
      if (Tiles.contains(reds, kind)) {
        redFive = kind;
      }
    }
    if ((left & ~Legal.swapBan(Meld.chi(base, called, Meld.AkaSource.NONE))) == 0) {
      return 0;
    }
    int added = 0;
    if (redFive != TurnEvent.NONE) {
      decisions.add(Action.chi(base, called, true));
      added++;
    }
    if (redFive == TurnEvent.NONE || hand.count(redFive) > 1) {
      decisions.add(Action.chi(base, called, false));
      added++;
    }
    return added;
  }

  /**
   * 書いている席の判断を書き終え、次に使う判断 ID を返す。ほかに選べる手があれば、書き終える前に見送りを足す。合法手が無ければ判断を
   * 残さず、ID も進めない。
   */
  private static long finish(Decisions decisions, int options, long id) {
    if (options > 0) {
      decisions.add(PASS_CODE);
    }
    return decisions.close(id) ? id + 1 : id;
  }

  /**
   * 席ごとに、選んだ手が行われたかと、選び替えで結果が変わりえたかを書く。選び替えは、他家の選んだ手をそのままにして自分だけ替えること。
   * ロンが合法な席は変わりえた。ロンで終わったなら、ロンが合法でない席は変えられない。誰も応じなかったか、自分の鳴きが通ったなら
   * 変わりえた。他家のチーが通ったなら、ポン・大明槓が合法な席だけが変えられた。変わりえた席のうち、合法手が 1 つしか無い席はそのことも書く。
   */
  private void resolve(Decisions decisions, int[] choices, Resolution resolution) {
    resolution.clear();
    boolean chiCalled = outcome == Outcome.CALL && Action.of(call).type() == Type.CHI;
    for (int slot = 0; slot < decisions.size(); slot++) {
      Type chosen = chosen(decisions, choices, slot);
      boolean own = outcome == Outcome.CALL && decisions.seat(slot) == caller;
      boolean executed =
          switch (outcome) {
            case PASS -> true;
            case CALL -> chosen == Type.PASS || own;
            case RON, TRIPLE_RON -> chosen == Type.PASS || chosen == Type.RON_AGARI;
          };
      boolean causal =
          has(decisions, slot, Type.RON_AGARI)
              || switch (outcome) {
                case PASS -> true;
                case CALL ->
                    own || chiCalled && (has(decisions, slot, Type.PON) || has(decisions, slot, Type.DAIMINKAN));
                case RON, TRIPLE_RON -> false;
              };
      Resolution.Role role =
          !causal
              ? Resolution.Role.PREEMPTED
              : decisions.legalCount(slot) == 1 ? Resolution.Role.FORCED : Resolution.Role.CAUSAL;
      resolution.add(decisions.id(slot), executed, role);
    }
  }

  /** ロンの無かった応答を締める。立直の供託を払い、見逃しの振聴を付ける。 */
  private void closeWithoutRon(PublicState state, Hand[] hands) {
    TurnEvent last = state.last();
    deposit(state, last.seat());
    missed(hands, last.seat(), last.tile());
  }

  /** 打牌が立直の宣言牌なら、供託を払って立直を成立させる。 */
  private static void deposit(PublicState state, int actor) {
    EventLog events = state.events();
    int last = events.size() - 1;
    if (events.kind(last) == EventKind.DISCARD && events.riichi(last)) {
      state.riichi(actor, events.riverSize(actor) == 1 && events.interruptions() == 0);
    }
  }

  /** 和了牌を見送った席に見逃しの印を立てる。 */
  private void missed(Hand[] hands, int actor, int kind) {
    for (int seat = 0; seat < seats; seat++) {
      if (seat != actor && Tiles.contains(hands[seat].waits(), kind)) {
        hands[seat].setMissedRon(true);
      }
    }
  }

  private void requireCall() {
    if (outcome != Outcome.CALL) {
      throw new IllegalStateException("The responses closed with " + outcome + ", not a call");
    }
  }

  private static Type chosen(Decisions decisions, int[] choices, int slot) {
    return Action.of(decisions.legal(slot, choices[slot])).type();
  }

  private static boolean has(Decisions decisions, int slot, Type type) {
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      if (Action.of(decisions.legal(slot, index)).type() == type) {
        return true;
      }
    }
    return false;
  }
}
