package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Random;

/**
 * テスト用の局。合法手の検査を省き、それらしい手順で状態の部品を進める。進めるのは、ツモ、暗槓・加槓・北抜きと嶺上ツモ、打牌と
 * 立直、ポン・チー・大明槓、槓ドラの即乗りと後めくり、判断と選んだ手の結果。和了は無く、局は山が尽きて終わる。ウォームアップの
 * 後はメモリを確保しない。
 */
final class RandomRound {

  final Variant variant;
  final Wall wall;
  final Hand[] hands;
  final PublicState state;
  final ChanceLog chance;
  final Decisions decisions;
  final Resolution resolution;

  private final TableState table;
  private final TileSet set;
  private final int seats;
  private int drawer;
  private int drawn;
  private boolean kanCall;
  private long nextId = 1;

  RandomRound(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    seats = variant.seats();
    wall = new Wall(variant);
    hands = new Hand[seats];
    for (int seat = 0; seat < seats; seat++) {
      hands[seat] = new Hand(variant);
    }
    state = new PublicState(variant);
    chance = new ChanceLog(variant);
    decisions = new Decisions(variant);
    resolution = new Resolution(variant);
    table = new TableState(variant);
  }

  /** 半荘を始め、最初の局を配る。 */
  void start(RoundStart start, long seed, int epoch, DrawOrder drawOrder) {
    wall.start(seed, epoch, drawOrder);
    deal(start);
  }

  /** 局の始まりの卓から、次の牌山で局を配り、最初のドラ表示牌をめくる。 */
  void deal(RoundStart start) {
    table.set(start);
    state.start(table);
    int dealer = state.dealer();
    wall.shuffle(dealer);
    for (Hand hand : hands) {
      hand.clear();
    }
    for (int offset = 0; offset < seats; offset++) {
      Hand hand = hands[(dealer + offset) % seats];
      for (int index = 0; index < Variant.HAND; index++) {
        hand.add(wall.dealt(offset, index));
      }
    }
    int indicator = wall.indicator(0);
    state.reveal(indicator);
    chance.begin(hands, indicator);
    decisions.clear();
    resolution.clear();
    drawer = dealer;
    drawn = TurnEvent.NONE;
    kanCall = false;
  }

  /** 手番を 1 つ進める。ツモから打牌まで進め、打牌が鳴かれれば、鳴かれなくなるまで鳴いた席の打牌を続ける。山が尽きていれば何もせず false を返す。 */
  boolean turn(Random random) {
    if (wall.liveLeft() == 0) {
      return false;
    }
    int seat = drawer;
    draw(seat, false);
    if (state.riichi(seat) == RiichiState.NONE) {
      kans(random, seat);
    }
    int caller = respond(random, seat, discard(random, seat));
    while (caller >= 0) {
      seat = caller;
      if (kanCall) {
        draw(seat, true);
      }
      caller = respond(random, seat, discard(random, seat));
    }
    drawer = (seat + 1) % seats;
    return true;
  }

  /** 山が尽きるまで手番を進める。 */
  void finish(Random random) {
    while (wall.liveLeft() > 0) {
      turn(random);
    }
  }

  /** 局末の裏ドラ表示牌を、めくったドラ表示牌の数だけ偶然のイベントに足す。 */
  void revealUra() {
    for (int index = 0; index < state.dora().count(); index++) {
      chance.appendUra(wall.ura(index));
    }
  }

  void copyFrom(RandomRound other) {
    wall.copyFrom(other.wall);
    for (int seat = 0; seat < seats; seat++) {
      hands[seat].copyFrom(other.hands[seat]);
    }
    state.copyFrom(other.state);
    chance.copyFrom(other.chance);
    decisions.copyFrom(other.decisions);
    resolution.copyFrom(other.resolution);
    table.copyFrom(other.table);
    drawer = other.drawer;
    drawn = other.drawn;
    kanCall = other.kanCall;
    nextId = other.nextId;
  }

  private void draw(int seat, boolean rinshan) {
    int tile = rinshan ? wall.drawRinshan(seat) : wall.draw(seat);
    hands[seat].add(tile);
    state.draw(seat, tile, rinshan);
    if (rinshan) {
      chance.appendRinshan(seat, tile);
    } else {
      chance.appendDraw(seat, tile);
    }
    drawn = tile;
  }

  /** 山が残っている間、ときどき暗槓・加槓・北抜きを続ける。暗槓の槓ドラはすぐめくり、加槓の槓ドラは後めくりにする。 */
  private void kans(Random random, int seat) {
    Hand hand = hands[seat];
    while (wall.liveLeft() > 0 && random.nextInt(3) == 0) {
      boolean kanLeft = state.kans() < Variant.MAX_KANS;
      int quad = kanLeft ? quad(hand) : -1;
      int pon = kanLeft ? upgradable(hand) : -1;
      if (quad >= 0) {
        for (int copy = 0; copy < Tiles.COPIES; copy++) {
          hand.take(Tiles.physicalId(quad, copy));
        }
        int meld = Meld.ankan(quad, set.red(Tiles.physicalId(quad, Tiles.RED_COPY))).id();
        hand.addMeld(meld);
        state.ankan(seat, meld);
        revealPending();
        revealIndicator();
      } else if (pon >= 0) {
        int kind = Meld.of(hand.meld(pon)).base();
        int added = held(hand, kind);
        hand.take(added);
        state.kakan(seat, hand.upgrade(kind, set.red(added)));
        revealPending();
        state.deferDora();
      } else if (variant.has(RuleFlag.NUKI) && hand.count(Tiles.PEI) > 0) {
        hand.take(held(hand, Tiles.PEI));
        hand.addNuki();
        state.nuki(seat);
        revealPending();
      } else {
        return;
      }
      draw(seat, true);
    }
  }

  /** 打牌する（立直中はツモ切り。ときどき立直を宣言する）。大明槓・加槓の槓ドラは打牌の後にめくる。打った牌の物理 id を返す。 */
  private int discard(Random random, int seat) {
    Hand hand = hands[seat];
    boolean inRiichi = state.riichi(seat) != RiichiState.NONE;
    boolean fromDraw = drawn != TurnEvent.NONE && (inRiichi || random.nextInt(4) == 0);
    int tile = fromDraw ? drawn : randomHeld(random, hand);
    boolean declares =
        !inRiichi && concealed(hand) && state.score(seat) >= Variant.DEPOSIT && random.nextInt(16) == 0;
    decisions.clear();
    decisions.open(seat, EngineDecisionKind.TURN);
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (hand.count(kind) > 0) {
        decisions.add(Action.code(Action.Type.DAHAI, kind, Action.TileSelection.NON_AKA));
      }
    }
    if (decisions.close(nextId)) {
      nextId++;
    }
    resolution.clear();
    Resolution.Role role = decisions.legalCount(0) == 1 ? Resolution.Role.FORCED : Resolution.Role.CAUSAL;
    resolution.add(decisions.id(0), true, role);
    hand.discard(tile);
    state.discard(seat, tile, tile == drawn, declares);
    drawn = TurnEvent.NONE;
    hand.setMissedRon(random.nextInt(8) == 0);
    if (declares) {
      state.riichi(seat, state.events().interruptions() == 0 && state.events().riverSize(seat) == 1);
    }
    revealPending();
    return tile;
  }

  /** 打牌への応答の判断を作り、ときどき大明槓・ポン・チーをする。鳴いた席を返し、鳴かなければ −1。 */
  private int respond(Random random, int discarder, int tile) {
    int kind = Tiles.kind(tile);
    decisions.clear();
    for (int offset = 1; offset < seats; offset++) {
      int seat = (discarder + offset) % seats;
      decisions.open(seat, EngineDecisionKind.RESPONSE);
      if (callable(seat, kind) && hands[seat].count(kind) >= 2) {
        decisions.add(Action.code(Action.Type.PON, kind, Action.TileSelection.NON_AKA));
      }
      decisions.add(Action.code(Action.Type.PASS));
      if (decisions.close(nextId)) {
        nextId++;
      }
    }
    int caller = -1;
    int meld = -1;
    for (int offset = 1; offset < seats && caller < 0; offset++) {
      int seat = (discarder + offset) % seats;
      if (!callable(seat, kind)) {
        continue;
      }
      int count = hands[seat].count(kind);
      if (count == Tiles.COPIES - 1 && state.kans() < Variant.MAX_KANS && random.nextInt(4) == 0) {
        meld = daiminkan(seat, kind);
      } else if (count >= 2 && random.nextInt(3) == 0) {
        meld = pon(random, seat, kind);
      } else if (offset == 1 && variant.has(RuleFlag.CHI) && random.nextInt(3) == 0) {
        meld = chi(random, seat, kind);
      }
      if (meld >= 0) {
        caller = seat;
      }
    }
    resolution.clear();
    for (int slot = 0; slot < decisions.size(); slot++) {
      boolean executed = caller < 0 || decisions.seat(slot) == caller;
      resolution.add(decisions.id(slot), executed, executed ? Resolution.Role.CAUSAL : Resolution.Role.PREEMPTED);
    }
    kanCall = caller >= 0 && Meld.of(meld).type().kan();
    if (caller >= 0) {
      state.call(caller, meld);
      state.clearIppatsu();
      if (kanCall) {
        state.deferDora();
      }
    }
    return caller;
  }

  private int daiminkan(int seat, int kind) {
    Hand hand = hands[seat];
    int meld = meldOf(Action.code(Action.Type.DAIMINKAN, kind, Action.TileSelection.NOT_APPLICABLE), seat);
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int physicalId = Tiles.physicalId(kind, copy);
      if (hand.holds(physicalId)) {
        hand.take(physicalId);
      }
    }
    hand.addMeld(meld);
    return meld;
  }

  private int pon(Random random, int seat, int kind) {
    Hand hand = hands[seat];
    int redId = Tiles.physicalId(kind, Tiles.RED_COPY);
    boolean handRed = set.red(redId) && hand.holds(redId);
    int plain = hand.count(kind) - (handRed ? 1 : 0);
    boolean useRed = handRed && (plain < 2 || random.nextBoolean());
    Action.TileSelection selection = useRed ? Action.TileSelection.AKA : Action.TileSelection.NON_AKA;
    int meld = meldOf(Action.code(Action.Type.PON, kind, selection), seat);
    hand.take(useRed ? redId : hand.pick(kind, false));
    hand.take(hand.pick(kind, false));
    hand.addMeld(meld);
    return meld;
  }

  /** 鳴いた牌を含む順子のうち、ほかの 2 枚を持つ最初のものでチーし、面子の番号を返す。無ければ −1。 */
  private int chi(Random random, int seat, int kind) {
    Tiles.Group suit = Tiles.Group.of(kind);
    if (!suit.suited()) {
      return -1;
    }
    Hand hand = hands[seat];
    for (int base = Math.max(suit.first(), kind - 2); base <= Math.min(kind, suit.last() - 2); base++) {
      int first = base == kind ? base + 1 : base;
      int second = base + 2 == kind ? base + 1 : base + 2;
      if (hand.count(first) == 0 || hand.count(second) == 0) {
        continue;
      }
      boolean usedRed = takeForChi(random, hand, first) | takeForChi(random, hand, second);
      int meld = meldOf(Action.chi(base, kind, usedRed), seat);
      hand.addMeld(meld);
      return meld;
    }
    return -1;
  }

  /**
   * 席が直前の打牌を鳴いてできる面子の番号を返す（赤の出どころはエンジンと同じ決め方）。大明槓は手の赤を見るので、牌を手から除く
   * 前に呼ぶ。
   *
   * @param code 鳴く手のコード
   * @param seat 鳴く席
   */
  private int meldOf(int code, int seat) {
    return Legal.meldOf(Action.of(code), seat, state.last(), hands[seat].redFives(), seats).id();
  }

  /** チーに使う 1 枚を手から除く。赤を使ったら true を返す。 */
  private boolean takeForChi(Random random, Hand hand, int kind) {
    int redId = Tiles.physicalId(kind, Tiles.RED_COPY);
    boolean handRed = set.red(redId) && hand.holds(redId);
    boolean useRed = handRed && (hand.count(kind) == 1 || random.nextBoolean());
    hand.take(useRed ? redId : hand.pick(kind, false));
    return useRed;
  }

  /** 鳴けるか。立直していない席は、面子にできる種類の打牌を鳴ける。山が尽きた後の最後の打牌は鳴けない。 */
  private boolean callable(int seat, int kind) {
    return wall.liveLeft() > 0
        && state.riichi(seat) == RiichiState.NONE
        && Tiles.contains(variant.meldableKinds(), kind);
  }

  private void revealPending() {
    if (state.dora().pending()) {
      revealIndicator();
    }
  }

  private void revealIndicator() {
    int tile = wall.indicator(state.dora().count());
    state.reveal(tile);
    chance.appendIndicator(tile);
  }

  /** 4 枚ある、面子にできる種類。無ければ −1。 */
  private int quad(Hand hand) {
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (hand.count(kind) == Tiles.COPIES && Tiles.contains(variant.meldableKinds(), kind)) {
        return kind;
      }
    }
    return -1;
  }

  /** 4 枚目を純手牌に持つポンの面子の位置。無ければ −1。 */
  private static int upgradable(Hand hand) {
    for (int index = 0; index < hand.meldCount(); index++) {
      Meld meld = Meld.of(hand.meld(index));
      if (meld.type() == Meld.Type.PON && hand.count(meld.base()) > 0) {
        return index;
      }
    }
    return -1;
  }

  private static boolean concealed(Hand hand) {
    for (int index = 0; index < hand.meldCount(); index++) {
      if (!Meld.of(hand.meld(index)).type().concealed()) {
        return false;
      }
    }
    return true;
  }

  /** 純手牌にあるその種類の牌のうち、最小の物理 id。 */
  private static int held(Hand hand, int kind) {
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int physicalId = Tiles.physicalId(kind, copy);
      if (hand.holds(physicalId)) {
        return physicalId;
      }
    }
    throw new IllegalStateException("No tile of kind " + kind);
  }

  private static int randomHeld(Random random, Hand hand) {
    while (true) {
      int kind = random.nextInt(Tiles.KINDS);
      if (hand.count(kind) > 0) {
        int start = random.nextInt(Tiles.COPIES);
        for (int step = 0; step < Tiles.COPIES; step++) {
          int physicalId = Tiles.physicalId(kind, (start + step) % Tiles.COPIES);
          if (hand.holds(physicalId)) {
            return physicalId;
          }
        }
      }
    }
  }
}
