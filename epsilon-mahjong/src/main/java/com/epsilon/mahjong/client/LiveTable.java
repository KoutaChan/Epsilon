package com.epsilon.mahjong.client;

import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatTable;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntPredicate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ライブの対局で、自分の席から見た卓。サーバが送るイベントを SeatTable に書き、自分の判断は Decider に選ばせる。席は起家を 0 と
 * する絶対席で、牌は物理 id で表す。
 *
 * <p>天鳳と RiichiLab は、立直の宣言を宣言牌より前に送る。大明槓・加槓の後めくりのドラ表示牌も、打牌より前に送る。そこで、立直の
 * 宣言はその席の次の打牌に付ける。表示牌は、SeatTable が受けられるようになるまで取っておく。
 *
 * <p>局の結果は、サーバの値（RoundResult）をこの卓でも精算して照らし合わせる。点の移動か精算後の卓が合わなければ、警告を出して
 * 数える。次の局は、サーバの卓から始める。同時に使えるスレッドは 1 つだけ。
 */
public final class LiveTable {

  public static final int NO_DECISION = -1;

  private static final Logger log = LoggerFactory.getLogger(LiveTable.class);

  private final Variant variant;
  private final SeatTable table;
  private final RuleWorkspace ws;
  private final Decider decider;
  private final RoundOutcome outcome;

  /** 受け取ったが、SeatTable がまだ受けられないドラ表示牌（受け取った順）。 */
  private final ArrayDeque<Integer> early = new ArrayDeque<>();

  private RoundStart start;

  /** 次の打牌が立直の宣言牌になる席（席ごとのビット）。 */
  private int declared;

  /** この局で立直が成立した席（席ごとのビット）。 */
  private int accepted;

  /** 自分がツモって、まだ打っていない牌。無ければ {@code TurnEvent.NONE}。 */
  private int drawn = TurnEvent.NONE;

  private int mismatches;

  public LiveTable(Variant variant, int seat, Decider decider) {
    this.variant = variant;
    table = new SeatTable(variant, seat);
    ws = new RuleWorkspace(variant);
    this.decider = Objects.requireNonNull(decider, "decider");
    outcome = new RoundOutcome(variant);
  }

  /** 自分の席。 */
  public int seat() {
    return table.seat();
  }

  /** 自分の席から見た卓。次のイベントまで有効。 */
  public SeatView view() {
    return table.view();
  }

  /** 自分の判断（判断が無ければ空）。次のイベントまで有効。 */
  public Decisions decisions() {
    return table.decisions();
  }

  /** サーバの局の結果が、この卓の精算と合わなかった局の数。 */
  public int mismatches() {
    return mismatches;
  }

  /**
   * 局を始める。
   *
   * @param start 局の始まり
   * @param hand 自分の配牌（物理 id 13 枚）
   * @param indicator 最初のドラ表示牌の物理 id
   */
  public void startRound(RoundStart start, int[] hand, int indicator) {
    table.startRound(start, hand, indicator);
    this.start = start;
    early.clear();
    declared = 0;
    accepted = 0;
    drawn = TurnEvent.NONE;
  }

  /**
   * ツモを受ける（嶺上牌を含む）。
   *
   * @param seat ツモった席
   * @param tile ツモった牌の物理 id。他家のツモなら {@code TurnEvent.NONE}
   */
  public void draw(int seat, int tile) {
    reveal();
    table.draw(seat, tile, ws);
    if (seat == table.seat()) {
      drawn = tile;
    }
  }

  /** 自分がツモって、まだ打っていない牌。無ければ {@code TurnEvent.NONE}。 */
  public int drawn() {
    return drawn;
  }

  /**
   * 打牌を受ける。立直を宣言した席の打牌なら、宣言牌にする。
   *
   * @param seat 打った席
   * @param tile 打った牌の物理 id
   * @param tsumogiri ツモ切りなら true
   */
  public void discard(int seat, int tile, boolean tsumogiri) {
    int bit = 1 << seat;
    table.discard(seat, tile, tsumogiri, (declared & bit) != 0, ws);
    declared &= ~bit;
    if (seat == table.seat()) {
      drawn = TurnEvent.NONE;
    }
    reveal();
  }

  /**
   * 直前の打牌へのチー・ポン・大明槓を受ける。
   *
   * @param seat 鳴いた席
   * @param meld 面子の番号
   */
  public void call(int seat, int meld) {
    table.call(seat, meld);
  }

  /**
   * 暗槓を受ける。
   *
   * @param seat 槓をした席
   * @param kind 槓にした牌の種類
   */
  public void closedKan(int seat, int kind) {
    table.closedKan(seat, kind);
    reveal();
  }

  /**
   * ポンに 4 枚目を足す加槓を受ける。
   *
   * @param seat 槓をした席
   * @param kind 槓にした牌の種類
   */
  public void addedKan(int seat, int kind) {
    table.addedKan(seat, kind, ws);
    reveal();
  }

  /**
   * 手番の席の北抜きを受ける。続いて来る保留の表示牌と補充のツモは、別のイベントとして受ける。
   *
   * @param seat 北を抜いた席
   * @param tile サーバが知らせた、抜いた北の物理 id。自分の北抜きでは、この北を手から除く
   */
  public void nuki(int seat, int tile) {
    table.nuki(seat, tile, ws);
    reveal();
  }

  /**
   * めくったドラ表示牌（物理 id）を受ける。SeatTable がまだ受けられなければ、受けられるようになるまで取っておく。自分の判断が出ている
   * 間も取っておき、次のツモの前に渡す。加槓か北抜きへの応答を求められているときに表示牌を渡すと、SeatTable はその応答を締めるので、
   * 応答を聞く前に表示牌を送るサーバでも、槍槓と北へのロンを選べるようにする。
   */
  public void dora(int tile) {
    early.addLast(tile);
    if (table.decisions().size() == 0) {
      reveal();
    }
  }

  /** 立直の宣言を受ける。その席の次の打牌が宣言牌になる。 */
  public void declareRiichi(int seat) {
    declared |= 1 << Objects.checkIndex(seat, variant.seats());
  }

  /** 立直の成立を受ける。宣言牌がロンされずに通ったときに呼ぶ。 */
  public void acceptRiichi(int seat) {
    table.riichiAccepted(seat);
    accepted |= 1 << seat;
  }

  /** この局で立直が成立した席（席ごとのビット）。 */
  public int riichiMask() {
    return accepted;
  }

  /**
   * 自分の判断があれば、合法手をサーバが出した手に絞って Decider に選ばせる。
   *
   * @param offered 行動の番号がサーバの出した手なら true を返す条件
   * @return 選んだ手の行動の番号。判断が無いとき、または絞った後に見送りのほかに手が残らないときは {@link #NO_DECISION}
   */
  public int decide(IntPredicate offered) {
    table.restrict(offered);
    Decisions decisions = table.decisions();
    if (decisions.size() == 0) {
      return NO_DECISION;
    }
    return decisions.legal(0, decider.decide(table.view(), decisions, 0));
  }

  /**
   * 行動で手から出す物理牌を書き出す（{@code SeatTable.tiles} と同じ）。
   *
   * @param code 自分の判断の合法手にある行動の番号
   * @param out 物理 id を書く先
   * @return 書いた枚数
   */
  public int tiles(int code, int[] out) {
    return table.tiles(code, out);
  }

  /**
   * サーバの局の結果をこの卓でも精算し、点の移動がサーバと同じで、精算後の卓が次の局の始まりと合うかを確かめる。合わなければ
   * 警告を出して数える。
   *
   * @param result サーバが送った局の結果
   * @param next サーバが始める次の局
   */
  public void checkRound(RoundResult result, RoundStart next) {
    RoundEnd end = settle(result);
    if (end == null) {
      return;
    }
    TableState after = end.next();
    if (!result.sameDeltas(end) || !after.startsAt(next)) {
      mismatch(result, "settles here to " + after + ", but the server starts " + next);
    }
  }

  /**
   * 半荘の最後の局の結果をこの卓でも精算し、点の移動と終局の点がサーバの値と合うかを確かめる。合わなければ警告を出して数える。
   *
   * @param result サーバが送った最後の局の結果
   * @param finalScores サーバが送った終局の点
   */
  public void checkGame(RoundResult result, int[] finalScores) {
    RoundEnd end = settle(result);
    if (end == null) {
      return;
    }
    TableState after = end.next();
    if (!result.sameDeltas(end) || !after.endsWith(finalScores)) {
      mismatch(result, "settles here to " + after + ", but the server ends with " + Arrays.toString(finalScores));
    }
  }

  /** 受けられるようになった表示牌を、受け取った順に SeatTable に渡す。 */
  private void reveal() {
    while (!early.isEmpty() && table.revealDue()) {
      table.revealDora(early.removeFirst(), ws);
    }
  }

  /** 局の結果をこの卓で精算する。精算できない結果（立直の席が違うなど）なら、警告を出して数え、null を返す。 */
  private RoundEnd settle(RoundResult result) {
    try {
      result.outcome(accepted, outcome);
      return table.settle(outcome);
    } catch (IllegalArgumentException | IllegalStateException failure) {
      mismatch(result, "cannot be settled here: " + failure.getMessage());
      return null;
    }
  }

  private void mismatch(RoundResult result, String detail) {
    mismatches++;
    log.warn("{}: the server ends the round with {}, which {}", start, result, detail);
  }
}
