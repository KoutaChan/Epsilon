package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * 1 席から見た卓。持つのは公開のイベントと自分の手牌だけで、他家の手については面子と抜いた北の数しか持たない。エンジンと同じ
 * 公開情報と手順（{@link Steps}）で書き換える。自分が判断するときはエンジンと同じ合法手を作り、合法手に無い自分の手は受けない。他家のツモは、
 * 引いた牌が分からないまま受ける。
 *
 * <p>イベントはエンジンが起こす順に受ける（牌譜の局の並び）。表示牌は、エンジンがめくる時点で受ける。めくるのは、暗槓の直後（保留の
 * 表示牌、この槓の表示牌の順）、明槓・加槓の後の打牌の直後、加槓への応答（槍槓）か抜いた北への応答が締まった直後（保留の表示牌）。
 * 打牌の後にめくる表示牌があれば、それを受けてから打牌への応答を求める。他家の応答は、続く鳴き・ツモ・表示牌を受けたときに、誰も
 * ロンしなかったとして締める。立直の成立を受けても締めない。ロンは局末の精算で受ける。裏ドラは分からないので、自分の和了の点に
 * 裏ドラは数えない。
 *
 * <p>配列はルールで決まる大きさで一度だけ確保し、イベントを受けるメソッドと局末の精算はメモリを確保しない。同時に使えるのは 1 つの
 * スレッドだけ。
 */
public final class SeatTable {

  /** 局の最初の判断 ID。 */
  private static final long FIRST_ID = 1;

  private static final int TSUMO_CODE = Action.code(Type.TSUMO_AGARI);
  private static final int RON_CODE = Action.code(Type.RON_AGARI);
  private static final int KYUSHU_CODE = Action.code(Type.KYUSHU_KYUHAI);
  private static final int PASS_CODE = Action.code(Type.PASS);
  private static final int NUKI_CODE = Action.code(Type.NUKI);

  /** めくるはずの表示牌を全て受けた後にすること。 */
  private enum AfterReveals {
    NOTHING,
    /** 打牌への応答を求める。 */
    OPEN_WINDOW,
    /** 加槓への応答（槍槓）が締まった後に、その槓の槓ドラを後めくりにする。 */
    DEFER_DORA
  }

  private final Variant variant;
  private final int seats;
  private final int self;
  private final Steps steps;
  private final PublicState state;
  private final Hand hand;
  private final Decisions decisions;
  private final SeatReader view;

  /** 局の始めの卓。精算はこの卓から行う。 */
  private final TableState table;

  private final RoundEnd end;
  private long nextId;

  /** 打牌・加槓・北抜きへの応答を求めている。次のイベントを受けたときに締める。 */
  private boolean windowOpen;

  /** 次のツモが嶺上牌なら、それを引く席。そうでなければ {@link TurnEvent#NONE}。 */
  private int replacement = TurnEvent.NONE;

  /** めくるはずで、まだ受けていない表示牌の数。 */
  private int revealsDue;

  private AfterReveals afterReveals = AfterReveals.NOTHING;

  public SeatTable(Variant variant, int seat) {
    this.variant = variant;
    seats = variant.seats();
    self = Objects.checkIndex(seat, seats);
    steps = new Steps(variant, 1 << seat);
    state = steps.state();
    hand = steps.hands()[seat];
    decisions = steps.decisions();
    view = new SeatReader(state, steps.hands(), decisions, seat);
    table = new TableState(variant);
    end = new RoundEnd(variant);
  }

  /** 見る席。 */
  public int seat() {
    return self;
  }

  /** 自分の判断。判断が無ければ空。次のイベントを受けるまで有効。 */
  public Decisions decisions() {
    return decisions;
  }

  /** 自分の席から見た卓のビュー。次のイベントを受けるまで有効。 */
  public SeatView view() {
    return view;
  }

  /**
   * 自分の合法手から、サーバが示さなかった手を除く。並びは保つ。手が残らないときと、見送りだけが残ったときは、判断ごと消す
   * （見送りは、ほかに選べる手があるときだけ合法手に入れる）。
   *
   * @param offered 行動の番号を受け、サーバが示した手なら true を返す条件
   */
  public void restrict(IntPredicate offered) {
    if (decisions.size() == 0) {
      return;
    }
    int kept = decisions.retain(0, offered);
    if (kept == 0 || kept == 1 && decisions.legal(0, 0) == PASS_CODE) {
      decisions.clear();
    }
  }

  /**
   * 自分の合法手を行うときに手から出す物理牌を書き、その枚数を返す。打牌・立直の打牌は打つ牌、チー・ポン・大明槓は手から面子へ出す牌、
   * 暗槓は 4 枚、加槓は足す牌、北抜きは抜く北。どれも、この卓がそのイベントを受けたときに手から除く牌と同じ。牌を出さない行動では 0。
   * 合法手に無い手には IllegalArgumentException を投げる。抜いた北を指定して受ける北抜きでは、ここで書く北でなく、指定した北を除く。
   *
   * @param code 行動の番号
   * @param out 物理 id を先頭から書く先
   * @return 書いた枚数
   */
  public int tiles(int code, int[] out) {
    requireLegal(code);
    return steps.tiles(self, Action.of(code), out);
  }

  /**
   * いま表示牌を受けられるなら true。めくるはずの表示牌が残っているときと、保留の槓ドラがあって加槓か北抜きへの応答を求めている
   * ときに限る。後のほうでは、表示牌を受けるとその応答を締める。
   */
  public boolean revealDue() {
    TurnEvent.Kind declared = state.last().kind();
    return revealsDue > 0
        || windowOpen && (declared == TurnEvent.Kind.KAN || declared == TurnEvent.Kind.NUKI) && state.dora().pending();
  }

  /**
   * 自分の配牌と、局の始めの表示牌で局を始める。
   *
   * @param start 局の始めの卓
   * @param ownHand 自分の配牌（物理 id 13 枚）
   * @param doraIndicator 局の始めの表示牌の物理 id
   */
  public void startRound(RoundStart start, int[] ownHand, int doraIndicator) {
    requireDeal(ownHand);
    table.set(start);
    begin(ownHand, doraIndicator);
  }

  /**
   * もしもの局を、指定した局・本場・供託・点と、自分の配牌・局の始めの表示牌で始める。値は範囲だけを確かめる（局の番号、0 以上の
   * 本場と供託、席の数だけの点）。点と供託の和が配給点の合計になることは求めない。ほかは {@link #startRound} で始めた局と同じように
   * 進め、局末の精算もこの値の卓から行う。範囲の外の値か、配牌が 13 枚でないときは、卓を変えずに IllegalArgumentException を投げる。
   *
   * @param round 局の通し番号
   * @param honba 本場
   * @param kyotaku 供託の本数
   * @param scores 絶対席の順の点（コピーする）
   * @param ownHand 自分の配牌（物理 id 13 枚）
   * @param indicator 局の始めの表示牌の物理 id
   */
  public void startScenario(int round, int honba, int kyotaku, int[] scores, int[] ownHand, int indicator) {
    requireDeal(ownHand);
    table.setScenario(round, honba, kyotaku, scores);
    begin(ownHand, indicator);
  }

  /**
   * ツモ（嶺上牌のツモを含む）を受ける。求めている応答があれば、誰もロンしなかったとして先に締める。他家のツモでは、引いた牌を
   * 読まない。自分のツモなら手番の判断を作る。ツモの番でない席のツモと、めくるはずの表示牌を受ける前のツモには
   * IllegalStateException を投げる。
   */
  public void draw(int seat, int tile, RuleWorkspace ws) {
    closeWindow();
    requireNoReveals();
    boolean rinshan = replacement != TurnEvent.NONE;
    int drawer = rinshan ? replacement : nextDrawer();
    if (seat != drawer) {
      throw new IllegalStateException("Seat " + seat + " draws out of turn, seat " + drawer + " draws next");
    }
    replacement = TurnEvent.NONE;
    decisions.clear();
    steps.draw(seat, seat == self ? tile : TurnEvent.NONE, rinshan);
    if (seat == self) {
      nextId = Legal.turn(state, hand, null, decisions, nextId, ws);
    }
  }

  /**
   * 打牌を受ける。打牌の後にめくる表示牌が無ければ、すぐに打牌への応答を求める。あれば、表示牌を受けてから求める。
   *
   * @param seat 打った席
   * @param tile 打った物理牌
   * @param tsumogiri ツモ切りなら true
   * @param riichi 立直の宣言牌なら true
   */
  public void discard(int seat, int tile, boolean tsumogiri, boolean riichi, RuleWorkspace ws) {
    requireQuiet();
    requireCurrent(seat);
    if (seat == self) {
      int code = Action.discard(Tiles.kind(tile), variant.tiles().red(tile), tsumogiri, riichi);
      requireLegal(code);
      steps.discard(seat, steps.tileOf(seat, Action.of(code)), tsumogiri, riichi);
    } else {
      steps.discard(seat, tile, tsumogiri, riichi);
    }
    decisions.clear();
    if (state.dora().pending()) {
      expectReveals(1, AfterReveals.OPEN_WINDOW);
    } else {
      openWindow(ws);
    }
  }

  /**
   * 直前の打牌への鳴きを受け、打牌への応答を締める。打牌への応答を求めていなければ IllegalStateException を投げる。
   *
   * @param seat 鳴いた席
   * @param meld チー・ポン・大明槓の面子の正準の番号
   */
  public void call(int seat, int meld) {
    if (!windowOpen || state.last().kind() != TurnEvent.Kind.DISCARD) {
      throw new IllegalStateException("No discard is open to a call by seat " + seat);
    }
    if (seat == self) {
      requireLegal(Action.call(Meld.of(meld)));
    }
    closeWindow();
    steps.call(seat, meld);
    if (Meld.of(meld).type() == Meld.Type.DAIMINKAN) {
      replacement = seat;
    } else if (seat == self) {
      nextId = Legal.postCall(hand, self, decisions, nextId);
    }
  }

  /**
   * 暗槓を受ける。この後は、保留の表示牌があればそれを、次にこの槓の表示牌を受け、最後に嶺上牌のツモを受ける。
   *
   * @param seat 暗槓した席
   * @param kind 槓にした種類
   */
  public void closedKan(int seat, int kind) {
    requireQuiet();
    requireCurrent(seat);
    if (seat == self) {
      requireLegal(Action.code(Type.ANKAN, kind, TileSelection.NOT_APPLICABLE));
    }
    int reveals = state.dora().pending() ? 2 : 1;
    decisions.clear();
    steps.ankan(seat, kind);
    expectReveals(reveals, AfterReveals.NOTHING);
    replacement = seat;
  }

  /**
   * 加槓を受け、加槓への応答（槍槓）を求める。
   *
   * @param seat 加槓した席
   * @param kind 加槓した種類
   */
  public void addedKan(int seat, int kind, RuleWorkspace ws) {
    requireQuiet();
    requireCurrent(seat);
    if (seat == self) {
      requireLegal(Action.code(Type.KAKAN, kind, TileSelection.NOT_APPLICABLE));
    }
    decisions.clear();
    steps.kakan(seat, kind);
    openWindow(ws);
    replacement = seat;
  }

  /**
   * 手番の席の北抜きを受け、抜いた北への応答を求める。この後は、保留の表示牌があればそれを受け、次に嶺上牌のツモを受ける。
   * 手番でない席の北抜きと、応答か表示牌を待っている間の北抜きには IllegalStateException を投げる。自分の北抜きが合法手に
   * 無ければ IllegalArgumentException を投げる。
   *
   * @param seat 北を抜いた席
   */
  public void nuki(int seat, RuleWorkspace ws) {
    requireQuiet();
    requireCurrent(seat);
    if (seat == self) {
      requireLegal(NUKI_CODE);
    }
    decisions.clear();
    steps.nuki(seat);
    openWindow(ws);
    replacement = seat;
  }

  /**
   * 手番の席の北抜きを、抜いた北の物理牌とともに受ける。自分の北抜きでは、卓は北を選ばずに、指定した北を手から除く。どの北を
   * 抜くかをサーバが決める対局で、自分の手の物理牌をサーバと合わせるために使う。ほかは {@link #nuki(int, RuleWorkspace)} と同じ。
   *
   * @param seat 北を抜いた席
   * @param tile 抜いた北の物理 id。他家の北抜きでは読まない
   * @throws IllegalArgumentException 自分の北抜きで、指定した牌が手にある北でない
   */
  public void nuki(int seat, int tile, RuleWorkspace ws) {
    if (seat != self) {
      nuki(seat, ws);
      return;
    }
    requireQuiet();
    requireCurrent(seat);
    requireLegal(NUKI_CODE);
    if (Tiles.kind(tile) != Tiles.PEI || !hand.holds(tile)) {
      throw new IllegalArgumentException("Seat " + self + " holds no north " + tile + " to take out");
    }
    decisions.clear();
    steps.nuki(seat, tile);
    openWindow(ws);
    replacement = seat;
  }

  /**
   * めくった表示牌を受ける。求めている応答があれば、誰もロンしなかったとして先に締める。めくるはずの表示牌が無ければ
   * IllegalStateException を投げる。
   *
   * @param tile 表示牌の物理 id
   */
  public void revealDora(int tile, RuleWorkspace ws) {
    closeWindow();
    if (revealsDue == 0) {
      throw new IllegalStateException("No dora indicator is due, but " + tile + " was revealed");
    }
    state.reveal(tile);
    revealsDue--;
    if (revealsDue > 0) {
      return;
    }
    AfterReveals after = afterReveals;
    afterReveals = AfterReveals.NOTHING;
    switch (after) {
      case OPEN_WINDOW -> openWindow(ws);
      case DEFER_DORA -> state.deferDora();
      case NOTHING -> {}
    }
  }

  /**
   * 立直の成立（宣言牌が通ったこと）を受ける。供託は、応答を締める次のイベントで払う。そのためここでは、求めている応答がその席の
   * 宣言牌への応答であるか、その席の立直が既に成り立っていることだけを確かめる。どちらでもなければ IllegalStateException を投げる。
   */
  public void riichiAccepted(int seat) {
    EventLog events = state.events();
    int last = events.size() - 1;
    boolean declared =
        windowOpen
            ? state.last().kind() == TurnEvent.Kind.DISCARD && events.seat(last) == seat && events.riichi(last)
            : state.riichi(seat) != RiichiState.NONE;
    if (!declared) {
      throw new IllegalStateException("Seat " + seat + " has no riichi to accept");
    }
  }

  /**
   * 局末を受け、局の結果を局の始めの卓から精算する。ロン（三家和を含む）で終わったのでなければ、求めている応答は、誰も応じなかった
   * として締める。
   *
   * @param outcome 局の結果。立直の席は、この局で供託を払った席
   * @return 局末。局の結果・点の動き・次の卓を持ち、和了の点と待ちは持たない。次の局を始めるまで有効
   * @throws IllegalArgumentException ルールの違う局の結果のとき、自分の和了か九種九牌が合法手に無いとき、立直の席がこの卓と違うとき
   */
  public RoundEnd settle(RoundOutcome outcome) {
    if (outcome.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + outcome.variant());
    }
    for (int i = 0; i < outcome.winnerCount(); i++) {
      if (outcome.winner(i) == self) {
        requireLegal(outcome.kind() == EndKind.TSUMO ? TSUMO_CODE : RON_CODE);
      }
    }
    if (outcome.abortive() == AbortiveDraw.NINE_TERMINALS && state.current() == self) {
      requireLegal(KYUSHU_CODE);
    }
    if (outcome.kind() == EndKind.RON || outcome.abortive() == AbortiveDraw.TRIPLE_RON) {
      windowOpen = false;
    } else {
      closeWindow();
    }
    decisions.clear();
    if (outcome.riichiMask() != state.riichiMask()) {
      throw new IllegalArgumentException(
          "Riichi seats " + Integer.toBinaryString(outcome.riichiMask()) + " differ from this table's "
              + Integer.toBinaryString(state.riichiMask()));
    }
    end.outcome().copyFrom(outcome);
    Settlement.apply(variant, table, outcome, end.delta());
    Progression.next(table, outcome, end.delta(), end.next());
    return end;
  }

  private static void requireDeal(int[] ownHand) {
    if (ownHand.length != Variant.HAND) {
      throw new IllegalArgumentException("A deal is " + Variant.HAND + " tiles, got " + ownHand.length);
    }
  }

  /** 局の始めの卓を書いた後に呼ぶ。局を始めて自分の配牌を入れ、局の始めの表示牌をめくる。 */
  private void begin(int[] ownHand, int doraIndicator) {
    steps.start(table);
    for (int tile : ownHand) {
      hand.add(tile);
    }
    state.reveal(doraIndicator);
    nextId = FIRST_ID;
    windowOpen = false;
    replacement = TurnEvent.NONE;
    revealsDue = 0;
    afterReveals = AfterReveals.NOTHING;
  }

  /**
   * 求めている応答を、誰もロンしなかったとして締める。宣言牌の立直があれば成立させて供託を払い、和了牌を見送った席に見逃しの印を
   * 付ける。加槓か北抜きへの応答なら全員の一発を消し、保留の表示牌があればそれを受けるのを待つ。加槓では、保留の表示牌があれば
   * それを受けた後に、無ければすぐに、この槓の槓ドラを後めくりにする。
   */
  private void closeWindow() {
    if (!windowOpen) {
      return;
    }
    windowOpen = false;
    decisions.clear();
    steps.pass();
    TurnEvent.Kind declared = state.last().kind();
    if (declared != TurnEvent.Kind.KAN && declared != TurnEvent.Kind.NUKI) {
      return;
    }
    state.clearIppatsu();
    boolean kan = declared == TurnEvent.Kind.KAN;
    if (state.dora().pending()) {
      expectReveals(1, kan ? AfterReveals.DEFER_DORA : AfterReveals.NOTHING);
    } else if (kan) {
      state.deferDora();
    }
  }

  private void openWindow(RuleWorkspace ws) {
    nextId = steps.respond(null, nextId, ws);
    windowOpen = true;
  }

  private void expectReveals(int count, AfterReveals after) {
    revealsDue = count;
    afterReveals = after;
  }

  /** 次に山からツモる席（嶺上牌は除く）。局の最初は親で、ほかは直前に打牌した席の次の席。 */
  private int nextDrawer() {
    if (state.events().size() == 0) {
      return state.dealer();
    }
    TurnEvent last = state.last();
    if (last.kind() != TurnEvent.Kind.DISCARD) {
      throw new IllegalStateException("No draw follows " + last.kind());
    }
    return (last.seat() + 1) % seats;
  }

  private void requireNoReveals() {
    if (revealsDue > 0) {
      throw new IllegalStateException(revealsDue + " dora indicators are due first");
    }
  }

  /** 応答を求めておらず、めくるはずの表示牌も残っていないことを確かめる。 */
  private void requireQuiet() {
    if (windowOpen) {
      throw new IllegalStateException("Responses are still pending");
    }
    requireNoReveals();
  }

  private void requireCurrent(int seat) {
    if (seat != state.current()) {
      throw new IllegalStateException("Seat " + seat + " acts out of turn, seat " + state.current() + " is on turn");
    }
  }

  /** 自分の合法手にその行動があることを確かめる。無ければ IllegalArgumentException を投げる。 */
  private void requireLegal(int code) {
    if (decisions.size() > 0) {
      for (int index = 0; index < decisions.legalCount(0); index++) {
        if (decisions.legal(0, index) == code) {
          return;
        }
      }
    }
    throw new IllegalArgumentException(Action.of(code) + " is not legal for seat " + self);
  }
}
