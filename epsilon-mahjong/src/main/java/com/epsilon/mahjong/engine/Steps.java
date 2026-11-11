package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 局の公開イベントを、公開情報と手に書き込む手順。エンジンの遷移と、1 席から見た卓（{@link SeatTable}）が同じ手順を使う。
 * 純手牌が分かる席（エンジンでは全席、1 席から見た卓では自分）は、手の牌を出し入れする。分からない席の手は、面子と抜いた北の数だけを
 * 持つ。その席は待ちが 0 なので、応答の判断は作られない。牌山と偶然のイベントは持たない。表示牌は呼び出し元が公開情報の側でめくる。
 * どのメソッドもメモリを確保しない。
 *
 * <p>槓: 暗槓は全員の一発を消し、表示牌は呼び出し元がすぐめくる。大明槓は全員の一発を消し、表示牌を後めくりにする。加槓では、
 * 加槓への応答（槍槓）が締まった後に、呼び出し元が一発を消して後めくりにする。後めくりの表示牌は、次の打牌の後（その打牌への応答の前）か、
 * 続く嶺上のツモの直前にめくる。
 *
 * <p>北抜き: 抜いた北は河に入らない。抜いた北への応答が締まった後に、呼び出し元が全員の一発を消し、保留の表示牌をめくってから
 * 補充の牌を引く。北抜きでは新しい表示牌をめくらない。抜いた数・見えている北・途切れは、加槓の牌と同じく宣言の時点で増やす。
 * ロンの点数は途切れを読まないので、応答の結果は増やすタイミングに依らない。
 *
 * <p>途中流局のうち四風連打・四家立直・四槓散了は、誰もロンしなかったら流れるかを、打牌への応答を求めるときにまとめて調べる。
 * 流れる打牌は鳴けない（ロンと見送りだけを選べる）。
 */
final class Steps {

  private final Variant variant;
  private final TileSet set;
  private final int seats;

  /** 純手牌が分かる席のマスク（ビット s が席 s）。 */
  private final int knownSeats;

  private final PublicState state;
  private final Hand[] hands;
  private final Decisions decisions;
  private final Window window;

  /**
   * 手番の席がこの手番にツモった物理 id（ツモ切りで打つ牌）。純手牌が分からない席のツモの後と、チー・ポン・加槓・北抜き・打牌の
   * 後は {@link TurnEvent#NONE}。
   */
  private int drawn = TurnEvent.NONE;

  /** 下家からポン・大明槓で鳴いた席が、まだ次の打牌をしていないなら true。その打牌では見逃しの印を消さない。 */
  private boolean reverseCall;

  /** いまの打牌に誰もロンしなかったら起きる途中流局。無ければ null。 */
  private AbortiveDraw abort;

  /** 鳴きで手から面子へ移す牌。{@link #calledTiles} がここに書く。 */
  private final int[] taken = new int[Tiles.COPIES];

  Steps(Variant variant, int knownSeats) {
    this.variant = variant;
    set = variant.tiles();
    seats = variant.seats();
    this.knownSeats = knownSeats;
    state = new PublicState(variant);
    hands = new Hand[seats];
    for (int seat = 0; seat < seats; seat++) {
      hands[seat] = new Hand(variant);
    }
    decisions = new Decisions(variant);
    window = new Window(variant);
  }

  PublicState state() {
    return state;
  }

  Hand[] hands() {
    return hands;
  }

  Decisions decisions() {
    return decisions;
  }

  Window window() {
    return window;
  }

  /** いまの打牌に誰もロンしなかったら起きる途中流局。無ければ null。 */
  AbortiveDraw abort() {
    return abort;
  }

  /** 卓の局を始め、手と判断を空にする。配牌と最初の表示牌は呼び出し元が入れる。 */
  void start(TableState table) {
    state.start(table);
    for (Hand hand : hands) {
      hand.clear();
    }
    decisions.clear();
    drawn = TurnEvent.NONE;
    reverseCall = false;
    abort = null;
  }

  /** ツモを書く。純手牌が分からない席では、引いた物理 id に {@link TurnEvent#NONE} を渡す。 */
  void draw(int seat, int tile, boolean rinshan) {
    if (known(seat)) {
      hands[seat].add(tile);
    } else if (tile != TurnEvent.NONE) {
      throw new IllegalArgumentException("Seat " + seat + " draws face down, not tile " + tile);
    }
    state.draw(seat, tile, rinshan);
    drawn = tile;
  }

  /**
   * 純手牌が分かる席が、打牌か立直の打牌で打つ物理 id。ツモ切りならツモった牌。そうでなければ、種類と赤かどうかが手の指定に合う
   * 牌のうち、ツモった牌を除いて物理 id が最も小さいもの。
   */
  int tileOf(int seat, Action action) {
    return action.selection().tsumogiri() ? drawn : hands[seat].pick(action.tile(), action.selection().red(), drawn);
  }

  /**
   * 純手牌が分かる席が、打牌か立直の打牌で打つ物理 id を、指定した牌で決める。
   *
   * @param tile 打つ物理 id。{@link TurnEvent#NONE} なら {@link #tileOf(int, Action)} で選ぶ
   * @throws IllegalArgumentException 指定した牌が、手にあって種類・赤かどうか・ツモ切りかが手の指定に合う牌でないとき
   */
  int tileOf(int seat, Action action, int tile) {
    if (tile == TurnEvent.NONE) {
      return tileOf(seat, action);
    }
    if (Tiles.kind(tile) != action.tile()
        || set.red(tile) != action.selection().red()
        || (tile == drawn) != action.selection().tsumogiri()
        || !hands[seat].holds(tile)) {
      throw new IllegalArgumentException("Seat " + seat + " cannot play tile " + tile + " for " + action);
    }
    return tile;
  }

  /**
   * 純手牌が分かる席が、北抜きで抜く物理 id を、指定した牌で決める。
   *
   * @param tile 抜く北の物理 id。{@link TurnEvent#NONE} なら、ツモった北か、純手牌の北のうち物理 id が最も小さいもの
   * @throws IllegalArgumentException 指定した牌が手にある北でないとき
   */
  int northOf(int seat, int tile) {
    if (tile == TurnEvent.NONE) {
      return nukiTile(hands[seat]);
    }
    if (Tiles.kind(tile) != Tiles.PEI || !hands[seat].holds(tile)) {
      throw new IllegalArgumentException("Seat " + seat + " holds no north " + tile + " to take out");
    }
    return tile;
  }

  /**
   * 純手牌が分かる席が手から出す物理牌を書き、枚数を返す。打牌・立直の打牌は打つ牌、チー・ポン・大明槓は {@link #call} が手から
   * 面子へ移す牌、暗槓はその種類の 4 枚、加槓は足す牌、北抜きは抜く北を書く。牌を出さない手は 0 を返す。
   *
   * @param seat 手を行う席
   * @param action 行う手
   * @param out 物理 id を先頭から書く先
   * @return 書いた枚数
   */
  int tiles(int seat, Action action, int[] out) {
    Hand hand = hands[seat];
    return switch (action.type()) {
      case DAHAI, RIICHI_DAHAI -> {
        out[0] = tileOf(seat, action);
        yield 1;
      }
      case CHI, PON, DAIMINKAN -> calledTiles(hand, Meld.of(meldOf(seat, action)), out);
      case ANKAN -> heldTiles(hand, action.tile(), out);
      case KAKAN -> {
        out[0] = addedTile(hand, action.tile());
        yield 1;
      }
      case NUKI -> {
        out[0] = nukiTile(hand);
        yield 1;
      }
      case TSUMO_AGARI, RON_AGARI, PASS, KYUSHU_KYUHAI -> 0;
    };
  }

  /**
   * 打牌する。純手牌が分かる席は手から牌を除き、見逃しの印を消す。ただし立直中と、下家から鳴いた後の打牌では消さない。上家・対面から
   * 鳴いた後の打牌（大明槓なら嶺上のツモの後の打牌）では消す。立直が成り立つのは宣言牌が通った後なので、立直の宣言牌でも消す。
   * 打った席の一発は消える。
   *
   * @param seat 打つ席
   * @param tile 打つ物理牌
   * @param tsumogiri ツモ切りなら true
   * @param riichi 立直の宣言牌なら true
   */
  void discard(int seat, int tile, boolean tsumogiri, boolean riichi) {
    if (known(seat)) {
      Hand hand = hands[seat];
      hand.discard(tile);
      if (!reverseCall && state.riichi(seat) == RiichiState.NONE) {
        hand.setMissedRon(false);
      }
    }
    reverseCall = false;
    state.discard(seat, tile, tsumogiri, riichi);
    state.clearIppatsu(seat);
    drawn = TurnEvent.NONE;
  }

  /**
   * 直前の打牌・加槓・北抜きへの応答を求め、応答する席ごとの判断を作る。打牌なら、誰もロンしなかったら起きる途中流局を先に調べる。
   *
   * @param wall 裏ドラ表示牌を読む牌山。1 席から見た卓では null
   * @param id 次に使う判断 ID
   * @param ws ロンの判定と点数の計算に使う作業領域
   * @return 判断を作った後に次に使う判断 ID
   */
  long respond(Wall wall, long id, RuleWorkspace ws) {
    abort = state.last().kind() == TurnEvent.Kind.DISCARD ? abortIfPassed() : null;
    return window.open(state, hands, wall, decisions, id, ws, abort == null);
  }

  /**
   * 判断する席の無い応答か、誰もロンしなかった応答を締める。宣言牌の立直があれば成立させて供託を払い、和了牌を見送った席に見逃しの
   * 印を付ける。
   */
  void pass() {
    window.pass(state, hands);
  }

  /**
   * 純手牌が分かる席が、直前の打牌をチー・ポン・大明槓で鳴いてできる面子の正準の番号。赤の由来は、鳴いた牌が赤なら鳴いた牌。
   * そうでなければ、手の赤の 5 を使うか（大明槓なら手の 3 枚に赤があるか）で決まる。
   */
  int meldOf(int caller, Action action) {
    return Legal.meldOf(action, caller, state.last(), hands[caller].redFives(), seats).id();
  }

  /**
   * 直前の打牌を、チー・ポン・大明槓の面子で鳴く。面子が直前の打牌を鳴いたものでなければ、手を変えずに例外を投げる。純手牌が分かる
   * 席は、鳴きに使う牌を手から除く。赤の由来が手の牌なら赤の 5 を 1 枚、ほかは赤でない牌のうち物理 id が最も小さいものを除く
   * （大明槓なら手の 3 枚を全て除く）。全員の一発を消し、大明槓なら表示牌を後めくりにする。
   *
   * @param caller 鳴く席
   * @param meldId 面子の正準の番号
   */
  void call(int caller, int meldId) {
    state.call(caller, meldId);
    Meld meld = Meld.of(meldId);
    Hand hand = hands[caller];
    if (known(caller)) {
      takeCalled(hand, meld);
    }
    hand.addMeld(meldId);
    state.clearIppatsu();
    reverseCall = meld.source() == Meld.RelativeSource.SHIMOCHA;
    if (meld.type() == Meld.Type.DAIMINKAN) {
      state.deferDora();
    }
  }

  /** その種類の 4 枚で暗槓し、全員の一発を消す。表示牌は呼び出し元がこの後めくる。 */
  void ankan(int seat, int kind) {
    Hand hand = hands[seat];
    int meld = Meld.ankan(kind, known(seat) ? takeAll(hand, kind) : redCopy(kind)).id();
    hand.addMeld(meld);
    state.ankan(seat, meld);
    state.clearIppatsu();
  }

  /**
   * その種類のポンに 4 枚目を足して加槓する。純手牌が分かる席は手の牌（赤があれば赤）を足す。分からない席の 4 枚目は残りの 1 枚
   * なので、その種類に赤の牌があり、ポンが赤を含まなければ赤とする。加槓への応答（槍槓）は呼び出し元が求める。
   */
  void kakan(int seat, int kind) {
    Hand hand = hands[seat];
    boolean addedRed;
    if (known(seat)) {
      int added = addedTile(hand, kind);
      hand.take(added);
      addedRed = set.red(added);
    } else {
      addedRed = redCopy(kind) && !redPon(hand, kind);
    }
    state.kakan(seat, hand.upgrade(kind, addedRed));
    drawn = TurnEvent.NONE;
  }

  /**
   * 手番の席が北を抜く。純手牌が分かる席は、ツモった北があればそれを、無ければ純手牌の北のうち物理 id が最も小さいものを除く。
   * 立直中に抜いた北が和了牌なら、ツモ和了を見送ったとして見逃しの印を立てる（立直中なので局の終わりまで残る）。立直が役になるので、
   * 形の上で待っている北ならツモ和了できた。抜いた北への応答は呼び出し元が求める。
   *
   * @param seat 北を抜く席
   */
  void nuki(int seat) {
    nuki(seat, known(seat) ? nukiTile(hands[seat]) : TurnEvent.NONE);
  }

  /**
   * 手番の席が、指定した北を抜く。どの北を手から除くかのほかは {@link #nuki(int)} と同じ。
   *
   * @param seat 北を抜く席
   * @param tile 純手牌が分かる席が手から除く北の物理 id。分からない席では読まない
   */
  void nuki(int seat, int tile) {
    Hand hand = hands[seat];
    boolean known = known(seat);
    if (known) {
      hand.take(tile);
    }
    hand.addNuki();
    if (known && state.riichi(seat) != RiichiState.NONE && Tiles.contains(hand.waits(), Tiles.PEI)) {
      hand.setMissedRon(true);
    }
    state.nuki(seat);
    drawn = TurnEvent.NONE;
  }

  void copyFrom(Steps other) {
    state.copyFrom(other.state);
    for (int seat = 0; seat < seats; seat++) {
      hands[seat].copyFrom(other.hands[seat]);
    }
    decisions.copyFrom(other.decisions);
    window.copyFrom(other.window);
    System.arraycopy(other.taken, 0, taken, 0, taken.length);
    drawn = other.drawn;
    reverseCall = other.reverseCall;
    abort = other.abort;
  }

  private boolean known(int seat) {
    return (knownSeats & 1 << seat) != 0;
  }

  /** 鳴きに使う手の牌を除く。 */
  private void takeCalled(Hand hand, Meld meld) {
    int count = calledTiles(hand, meld, taken);
    for (int index = 0; index < count; index++) {
      hand.take(taken[index]);
    }
  }

  /**
   * チー・ポン・大明槓で手から面子へ移す牌を書き、枚数を返す。赤の由来が手の牌なら赤の 5 を 1 枚、ほかは赤でない牌のうち物理 id が
   * 最も小さいものを選ぶ（大明槓なら手にあるその種類の全て）。
   */
  private static int calledTiles(Hand hand, Meld meld, int[] out) {
    int called = meld.called();
    boolean handRed = meld.aka() == Meld.AkaSource.CONSUMED_HAND_TILE;
    return switch (meld.type()) {
      case PON -> {
        out[0] = hand.pick(called, handRed);
        out[1] = hand.pick(called, false, out[0]);
        yield 2;
      }
      case DAIMINKAN -> heldTiles(hand, called, out);
      case CHI -> {
        int count = 0;
        for (int run = meld.base(); run < meld.base() + Meld.Type.CHI.tiles(); run++) {
          if (run != called) {
            out[count++] = hand.pick(run, handRed && Tiles.contains(Tiles.FIVES, run));
          }
        }
        yield count;
      }
      case ANKAN, KAKAN -> throw new IllegalArgumentException("Not a call: " + meld);
    };
  }

  /** 純手牌にあるその種類の物理牌を全て、物理 id の順に書き、枚数を返す。 */
  private static int heldTiles(Hand hand, int kind, int[] out) {
    int count = 0;
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(kind, copy);
      if (hand.holds(tile)) {
        out[count++] = tile;
      }
    }
    return count;
  }

  /** 加槓でポンに足す牌。純手牌に赤があれば赤で、ほかは物理 id が最も小さい牌。 */
  private static int addedTile(Hand hand, int kind) {
    return hand.pick(kind, Tiles.contains(hand.redFives(), kind));
  }

  /** 手番の席が抜く北。この手番にツモった北があればそれで、無ければ純手牌の北のうち物理 id が最も小さいもの。 */
  private int nukiTile(Hand hand) {
    return drawn != TurnEvent.NONE && Tiles.kind(drawn) == Tiles.PEI ? drawn : hand.pick(Tiles.PEI, false);
  }

  /**
   * いまの打牌に誰もロンしなかったら起きる途中流局。四風連打は、鳴きや槓で途切れていない第一巡で、全員の最初の打牌が同じ風のとき。
   * 四家立直は、ほかの全員が立直しているときの宣言牌。四槓散了は、2 人以上の槓が合わせて 4 つになった後の打牌。どれでもなければ null。
   */
  private AbortiveDraw abortIfPassed() {
    EventLog events = state.events();
    if (variant.aborts(AbortiveDraw.FOUR_WINDS) && sameWindRound(events)) {
      return AbortiveDraw.FOUR_WINDS;
    }
    if (variant.aborts(AbortiveDraw.FOUR_RIICHI)
        && events.riichi(events.size() - 1)
        && Integer.bitCount(state.riichiMask()) == seats - 1) {
      return AbortiveDraw.FOUR_RIICHI;
    }
    if (variant.aborts(AbortiveDraw.FOUR_KANS) && state.kans() == Variant.MAX_KANS && !oneSeatHoldsEveryKan()) {
      return AbortiveDraw.FOUR_KANS;
    }
    return null;
  }

  /** 鳴き・槓の無いまま全員が 1 枚ずつ打ち、どれも同じ風なら true。 */
  private boolean sameWindRound(EventLog events) {
    if (events.interruptions() > 0 || events.dahai() != seats) {
      return false;
    }
    int wind = events.tile(events.river(0, 0));
    if (Tiles.Group.of(wind) != Tiles.Group.WIND) {
      return false;
    }
    for (int seat = 1; seat < seats; seat++) {
      if (events.tile(events.river(seat, 0)) != wind) {
        return false;
      }
    }
    return true;
  }

  private boolean oneSeatHoldsEveryKan() {
    for (Hand hand : hands) {
      int kans = 0;
      for (int index = 0; index < hand.meldCount(); index++) {
        if (Meld.of(hand.meld(index)).type().kan()) {
          kans++;
        }
      }
      if (kans == state.kans()) {
        return true;
      }
    }
    return false;
  }

  /**
   * その種類の 4 枚に赤の 5 があるなら true。どの牌の集合でも、赤の 5 は種類の中で決まった番号
   * （{@code Tiles.RED_COPY}）の 1 枚なので、その牌だけを見る。
   */
  private boolean redCopy(int kind) {
    return set.red(Tiles.physicalId(kind, Tiles.RED_COPY));
  }

  /** 手にあるその種類のポンが赤を含むなら true。ポンが無ければ IllegalStateException を投げる。 */
  private static boolean redPon(Hand hand, int kind) {
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = Meld.of(hand.meld(index));
      if (meld.type() == Meld.Type.PON && meld.base() == kind) {
        return meld.red();
      }
    }
    throw new IllegalStateException("No pon of kind " + kind + " to upgrade");
  }

  /** 手にあるその種類の牌を全て除き、赤の 5 を含んでいたら true を返す。 */
  private boolean takeAll(Hand hand, int kind) {
    boolean red = false;
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(kind, copy);
      if (hand.holds(tile)) {
        red |= set.red(tile);
        hand.take(tile);
      }
    }
    return red;
  }
}
