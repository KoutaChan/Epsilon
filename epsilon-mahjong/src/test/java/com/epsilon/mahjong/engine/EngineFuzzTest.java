package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Factory;
import org.testng.annotations.Test;

/**
 * 自己対局（四人打ちと三人打ち）の全ての判断の時点で、状態を数え直した値と照合する。どの物理牌もちょうど 1 か所にある。点と供託の
 * 合計は変わらず、打牌の数は上限を超えない。見えている枚数・誰にも見えていない枚数・コピーした手・待ち・河にある牌の種類・立直者の
 * 現物・抜いた北の数は、数え直した値と合う。どの席のビューでも、手番の席は判断と合う。合法手は決まった順に並び、重複しない。判断の
 * 番号は 1 から始まり 1 ずつ増える。局の結果（終わり方・和了者・放銃者・包）は、局を終わらせた選択と合う。局末の点の移動と次の卓は、
 * 局の結果を {@code Settlement.apply} と {@code Progression.next} に渡した値と合う。
 *
 * <p>打ち手はランダムに選ぶ。和了・槓・北抜き・鳴き・立直が選べれば決まった確率で選び、打牌は向聴数がいちばん小さくなる牌から選ぶ。
 * 和了と途中流局を十分な数だけ起こすためである。
 */
public class EngineFuzzTest {

  private static final int GAMES = 200;

  private static final long DRAGONS = Tiles.range(Tiles.HAKU, Tiles.CHUN);
  private static final long WINDS = Tiles.range(Tiles.TON, Tiles.PEI);

  /** 打牌が通った後に起きる途中流局。 */
  private static final Set<AbortiveDraw> AFTER_DISCARD =
      EnumSet.of(AbortiveDraw.FOUR_WINDS, AbortiveDraw.FOUR_RIICHI, AbortiveDraw.FOUR_KANS);

  /** 物理牌ごとに、見つかった置き場の数。数えるのは手・牌山・ドラ表示牌で、河・面子・抜いた北は数えない。 */
  private final int[] places = new int[Tiles.PHYSICAL_IDS];

  private final Variant variant;
  private final int seats;
  private final Engine engine;
  private final RuleWorkspace ws;
  private final ShapeAnalyzer analyzer;
  private final HandShape shape = new HandShape();

  /** FullView がコピーした手。 */
  private final HandShape copied = new HandShape();

  private final Wall wall;
  private final TableState start;
  private final Map<String, Integer> coverage = new TreeMap<>();

  /** 直前に確定した判断で、判断した席と選んだ手の種類。先頭から、判断した席の数だけが有効。 */
  private final int[] pickedSeats;
  private final Action.Type[] picked;

  private long nextId;
  private int pickedCount;

  /** 直前の判断のきっかけになったイベントを起こした席。打牌・加槓・北抜きへの応答なら、放銃しうる席。 */
  private int actor;

  /** いまの半荘が席別ツモ列を使うか。 */
  private boolean streams;

  /** いまの局で北を抜かない席のビットマスク。北を手に残して北待ちの手を作り、抜いた北への応答を起こすため。 */
  private int keepNorths;

  @Factory(dataProvider = "variants")
  public EngineFuzzTest(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    engine = new Engine(variant);
    ws = new RuleWorkspace(variant);
    analyzer = new ShapeAnalyzer(variant);
    wall = (Wall) Fields.read(Fields.read(engine, "round"), "wall");
    start = new TableState(variant);
    pickedSeats = new int[seats - 1];
    picked = new Action.Type[seats - 1];
  }

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test
  public void everyBoundaryMatchesARecount() {
    Random random = new Random(7);
    for (int game = 0; game < GAMES; game++) {
      streams = game % 2 == 0;
      engine.startGame(random.nextLong(), streams ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
      start.copyFrom(new TableState(variant));
      nextId = 1;
      Boundary boundary = deal(random);
      while (true) {
        checkState();
        if (boundary == Boundary.DECISIONS) {
          boundary = decide(random);
          continue;
        }
        checkRoundEnd();
        if (boundary == Boundary.GAME_END) {
          break;
        }
        start.copyFrom(engine.roundEnd().next());
        boundary = deal(random);
      }
    }
    List<String> expected =
        variant.has(RuleFlag.NUKI)
            ? List.of(
                "TSUMO", "RON", "EXHAUSTIVE", "NINE_TERMINALS", "PON", "DAIMINKAN", "ANKAN", "KAKAN", "NUKI",
                "NUKI_RESPONSE", "RON_ON_NUKI", "RIICHI_DAHAI")
            : List.of(
                "TSUMO", "RON", "EXHAUSTIVE", "NINE_TERMINALS", "CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN",
                "RIICHI_DAHAI");
    for (String seen : expected) {
      Assert.assertTrue(coverage.getOrDefault(seen, 0) > 0, variant + ": " + seen + " never happened: " + coverage);
    }
  }

  /** 次の局を配る。北を抜くルールでは、この局で北を抜かない席を選ぶ。 */
  private Boundary deal(Random random) {
    keepNorths = variant.has(RuleFlag.NUKI) ? random.nextInt(1 << seats) : 0;
    return engine.advance(ws);
  }

  /** 判断する席ごとに、合法手の並びと判断の番号を確かめてから手を選ぶ。手を確定した後は、結果が判断と合うことを確かめる。 */
  private Boundary decide(Random random) {
    Decisions decisions = engine.decisions();
    int size = decisions.size();
    EngineDecisionKind kind = decisions.kind(0);
    long[] ids = new long[size];
    int[] counts = new int[size];
    int[] choices = new int[size];
    actor = engine.view(decisions.seat(0)).last().seat();
    int current =
        kind == EngineDecisionKind.TURN || kind == EngineDecisionKind.POST_CALL_DAHAI ? decisions.seat(0) : actor;
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(engine.view(seat).current(), current, "the seat on turn at " + kind);
    }
    pickedCount = size;
    for (int slot = 0; slot < size; slot++) {
      Assert.assertEquals(decisions.id(slot), nextId++, "decision ids run from 1 without gaps");
      Assert.assertEquals(engine.view(decisions.seat(slot)).kind(), decisions.kind(slot));
      count(decisions.kind(slot).name());
      checkOrder(decisions, slot);
      ids[slot] = decisions.id(slot);
      counts[slot] = decisions.legalCount(slot);
      choices[slot] = choose(decisions, slot, random);
      pickedSeats[slot] = decisions.seat(slot);
      picked[slot] = Action.of(decisions.legal(slot, choices[slot])).type();
      count(picked[slot].name());
    }
    Boundary boundary = engine.commit(choices, ws);
    Resolution resolution = engine.resolution();
    Assert.assertEquals(resolution.size(), size);
    for (int slot = 0; slot < size; slot++) {
      Assert.assertEquals(resolution.id(slot), ids[slot]);
      if (kind == EngineDecisionKind.TURN || kind == EngineDecisionKind.POST_CALL_DAHAI) {
        Assert.assertTrue(resolution.executed(slot));
        Assert.assertEquals(
            resolution.role(slot), counts[slot] == 1 ? Resolution.Role.FORCED : Resolution.Role.CAUSAL);
      }
    }
    return boundary;
  }

  /**
   * 和了・九種九牌・槓・北抜き・鳴き・立直が選べれば、決まった確率で選ぶ。どれも選ばなければ、見送りか、向聴数がいちばん小さくなる
   * 打牌を選ぶ。8 回に 1 回は合法手から一様に選ぶ。
   */
  private int choose(Decisions decisions, int slot, Random random) {
    int count = decisions.legalCount(slot);
    if (random.nextInt(8) == 0) {
      return random.nextInt(count);
    }
    int win = -1;
    int kyushu = -1;
    int kan = -1;
    int nuki = -1;
    int call = -1;
    int pass = -1;
    int riichi = -1;
    int riichis = 0;
    for (int index = 0; index < count; index++) {
      switch (Action.of(decisions.legal(slot, index)).type()) {
        case TSUMO_AGARI, RON_AGARI -> win = index;
        case KYUSHU_KYUHAI -> kyushu = index;
        case ANKAN, KAKAN -> kan = kan < 0 ? index : kan;
        case NUKI -> nuki = index;
        case CHI, PON, DAIMINKAN -> call = call < 0 ? index : call;
        case PASS -> pass = index;
        case RIICHI_DAHAI -> {
          riichi = riichi < 0 ? index : riichi;
          riichis++;
        }
        default -> {}
      }
    }
    if (win >= 0 && random.nextInt(8) > 0) {
      return win;
    }
    if (kyushu >= 0 && random.nextBoolean()) {
      return kyushu;
    }
    if (kan >= 0 && random.nextInt(3) == 0) {
      return kan;
    }
    if (nuki >= 0 && (keepNorths & 1 << decisions.seat(slot)) == 0 && random.nextInt(4) > 0) {
      return nuki;
    }
    if (call >= 0 && random.nextInt(4) == 0) {
      return call;
    }
    if (pass >= 0) {
      return pass;
    }
    if (riichis > 0 && random.nextBoolean()) {
      return riichi + random.nextInt(riichis);
    }
    return bestDiscard(decisions, slot, random);
  }

  private int bestDiscard(Decisions decisions, int slot, Random random) {
    load(engine.view(decisions.seat(slot)), decisions.seat(slot));
    int best = Integer.MAX_VALUE;
    int chosen = -1;
    int ties = 0;
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      Action action = Action.of(decisions.legal(slot, index));
      if (action.type() != Action.Type.DAHAI) {
        continue;
      }
      shape.remove(action.tile());
      int shanten = analyzer.shanten(shape);
      shape.add(action.tile());
      if (shanten < best) {
        best = shanten;
        chosen = index;
        ties = 1;
      } else if (shanten == best && random.nextInt(++ties) == 0) {
        chosen = index;
      }
    }
    return chosen;
  }

  /** 点・枚数・待ち・河・現物・物理牌を、イベントと手から数え直して照合する。 */
  private void checkState() {
    FullView full = engine.full();
    SeatView table = full.seat(0);
    EventLog events = table.events();
    int total = Variant.DEPOSIT * table.kyotaku();
    for (int seat = 0; seat < seats; seat++) {
      total += table.score(seat);
    }
    Assert.assertEquals(total, seats * variant.startPoints(), "points and deposits");
    Assert.assertTrue(events.dahai() <= variant.maxDiscards(), events.dahai() + " discards");

    int[] visible = new int[Tiles.KINDS];
    for (int event = 0; event < events.size(); event++) {
      switch (events.kind(event)) {
        case DISCARD -> visible[events.tile(event)]++;
        case MELD -> {
          Meld meld = Meld.of(events.meld(event));
          for (int index = 0; index < meld.size(); index++) {
            visible[meld.tile(index)]++;
          }
          if (meld.source() != Meld.RelativeSource.NONE) {
            visible[meld.called()]--;
          }
        }
        case KAKAN -> visible[Meld.of(events.meld(event)).base()]++;
        case NUKI -> visible[Tiles.PEI]++;
      }
    }
    for (int index = 0; index < table.indicatorCount(); index++) {
      visible[Tiles.kind(table.indicator(index))]++;
    }
    ChanceLog chance = full.chance();
    int[] ura = new int[Tiles.KINDS];
    for (int index = 0; index < chance.size(); index++) {
      if (chance.kind(index) == ChanceKind.URA) {
        ura[Tiles.kind(chance.tile(index))]++;
      }
    }
    TileSet set = variant.tiles();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(table.visible(kind), visible[kind], "visible " + kind);
      int held = 0;
      for (int seat = 0; seat < seats; seat++) {
        held += full.seat(seat).count(kind);
      }
      Assert.assertEquals(full.unseen(kind), set.copies(kind) - held - visible[kind] - ura[kind], "unseen " + kind);
    }

    for (int seat = 0; seat < seats; seat++) {
      load(full.seat(seat), seat);
      full.copyHand(seat, copied);
      Fields.assertSameFields(shape, copied);
      long waits = shape.size() == Variant.HAND ? analyzer.waits(shape) : 0;
      Assert.assertEquals(full.waits(seat), waits, "waits of seat " + seat);
      long river = 0;
      for (int index = 0; index < events.riverSize(seat); index++) {
        river |= Tiles.bit(events.tile(events.river(seat, index)));
      }
      Assert.assertEquals(table.riverKinds(seat), river, "river of seat " + seat);
      Assert.assertEquals(table.riichiSafe(seat), riichiSafe(table, seat), "riichi-safe kinds of seat " + seat);
      Assert.assertEquals(table.nuki(seat), nukis(events, seat), "nuki of seat " + seat);
    }
    checkTiles(full);
  }

  /** その席が北を抜いたイベントの数。 */
  private static int nukis(EventLog events, int seat) {
    int count = 0;
    for (int event = 0; event < events.size(); event++) {
      count += events.kind(event) == EventKind.NUKI && events.seat(event) == seat ? 1 : 0;
    }
    return count;
  }

  /** その席の立直の宣言牌より後に打たれた牌の種類（ビットの集合）。立直していなければ 0。 */
  private static long riichiSafe(SeatView table, int seat) {
    if (table.riichi(seat) == RiichiState.NONE) {
      return 0;
    }
    EventLog events = table.events();
    long safe = 0;
    boolean declared = false;
    for (int event = 0; event < events.size(); event++) {
      if (events.kind(event) != EventKind.DISCARD) {
        continue;
      }
      if (declared) {
        safe |= Tiles.bit(events.tile(event));
      }
      declared |= events.seat(event) == seat && events.riichi(event);
    }
    return safe;
  }

  /**
   * 物理牌の置き場を照合する。手・牌山のまだ読まれていない位置・めくったドラ表示牌に、どの物理牌も 2 回以上は現れない。種類
   * ごとに赤と赤でない牌を分けて、そこにある数に河（鳴かれていない打牌）・面子・抜いた北の数を足すと、ルールの牌の集合にある枚数に
   * なる。k 回目の嶺上ツモが読んだ位置は、席別ツモ列の局なら王牌の始まりの k 枚手前。そうでなければ王牌の嶺上の k 枚目で、4 回を
   * 超えた k 回目は王牌の始まりの k − 4 枚手前。
   */
  private void checkTiles(FullView full) {
    TileSet set = variant.tiles();
    SeatView table = full.seat(0);
    Arrays.fill(places, 0);
    for (int seat = 0; seat < seats; seat++) {
      for (int index = 0; index < set.size(); index++) {
        int tile = set.physicalId(index);
        if (full.seat(seat).holds(tile)) {
          places[tile]++;
        }
      }
    }
    for (int index = 0; index < table.indicatorCount(); index++) {
      places[table.indicator(index)]++;
    }
    ChanceLog chance = full.chance();
    int draws = 0;
    int rinshans = 0;
    for (int index = 0; index < chance.size(); index++) {
      draws += chance.kind(index) == ChanceKind.DRAW ? 1 : 0;
      rinshans += chance.kind(index) == ChanceKind.RINSHAN ? 1 : 0;
    }
    int size = set.size();
    int deadStart = size - Variant.DEAD_WALL;
    for (int position = Variant.HAND * seats + draws; position < size; position++) {
      boolean read = false;
      for (int k = 1; k <= rinshans; k++) {
        int rinshan =
            streams
                ? deadStart - k
                : k <= Variant.MAX_KANS ? size - k : deadStart - (k - Variant.MAX_KANS);
        read |= position == rinshan;
      }
      for (int index = 0; index < table.indicatorCount(); index++) {
        read |= position == size - Variant.MAX_KANS - 2 - 2 * index;
      }
      if (!read) {
        places[wall.tile(position)]++;
      }
    }
    int[] publicRed = new int[Tiles.KINDS];
    int[] publicPlain = new int[Tiles.KINDS];
    EventLog events = table.events();
    for (int event = 0; event < events.size(); event++) {
      if (events.kind(event) == EventKind.DISCARD && events.calledBy(event) == EventLog.NONE) {
        (events.red(event) ? publicRed : publicPlain)[events.tile(event)]++;
      }
    }
    for (int seat = 0; seat < seats; seat++) {
      publicPlain[Tiles.PEI] += table.nuki(seat);
      for (int index = 0; index < table.meldCount(seat); index++) {
        Meld meld = Meld.of(table.meld(seat, index));
        boolean red = meld.red();
        for (int tile = 0; tile < meld.size(); tile++) {
          int kind = meld.tile(tile);
          if (red && Tiles.contains(Tiles.FIVES, kind)) {
            publicRed[kind]++;
            red = false;
          } else {
            publicPlain[kind]++;
          }
        }
      }
    }
    for (long kinds = set.kinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      int red = 0;
      int plain = 0;
      int redCopies = 0;
      for (int copy = 0; copy < Tiles.COPIES; copy++) {
        int tile = Tiles.physicalId(kind, copy);
        Assert.assertTrue(places[tile] <= 1, "tile " + tile + " is in " + places[tile] + " places");
        redCopies += set.red(tile) ? 1 : 0;
        if (set.red(tile)) {
          red += places[tile];
        } else {
          plain += places[tile];
        }
      }
      Assert.assertEquals(red + publicRed[kind], redCopies, "red tiles of kind " + kind);
      Assert.assertEquals(plain + publicPlain[kind], Tiles.COPIES - redCopies, "plain tiles of kind " + kind);
    }
  }

  /**
   * 合法手の並びを確かめる。並べるキー（手の種類の順、牌の種類か面子の位置、物理牌の選び方の順）が、前の手より必ず大きい。応答の判断は
   * 見送りで終わり、見送りしか無い応答の判断は無い。
   */
  private void checkOrder(Decisions decisions, int slot) {
    int count = decisions.legalCount(slot);
    long previous = Long.MIN_VALUE;
    for (int index = 0; index < count; index++) {
      Action action = Action.of(decisions.legal(slot, index));
      long key = key(decisions.kind(slot), action, decisions.seat(slot));
      Assert.assertTrue(key > previous, "legal actions out of order at " + action + " in slot " + slot);
      previous = key;
    }
    EngineDecisionKind kind = decisions.kind(slot);
    if (kind != EngineDecisionKind.TURN && kind != EngineDecisionKind.POST_CALL_DAHAI) {
      Assert.assertTrue(count > 1, "a response offers more than a pass");
      Assert.assertEquals(Action.of(decisions.legal(slot, count - 1)).type(), Action.Type.PASS);
    }
  }

  private long key(EngineDecisionKind kind, Action action, int seat) {
    int selection =
        switch (action.selection()) {
          case NOT_APPLICABLE, AKA -> 0;
          case NON_AKA -> 1;
          case TSUMOGIRI, TSUMOGIRI_AKA -> 2;
        };
    int rank;
    int sub = action.tile();
    switch (kind) {
      case TURN, POST_CALL_DAHAI -> {
        rank =
            switch (action.type()) {
              case TSUMO_AGARI -> 0;
              case ANKAN -> 1;
              case KAKAN -> 2;
              case NUKI -> 3;
              case RIICHI_DAHAI -> 4;
              case DAHAI -> 5;
              case KYUSHU_KYUHAI -> 6;
              default -> throw new AssertionError(action + " in a " + kind + " slot");
            };
        if (action.type() == Action.Type.KAKAN) {
          sub = ponIndex(seat, action.tile());
        }
      }
      case RESPONSE, CHANKAN, NUKI_RESPONSE -> {
        rank =
            switch (action.type()) {
              case RON_AGARI -> 0;
              case PON -> 1;
              case DAIMINKAN -> 2;
              case CHI -> 3;
              case PASS -> 4;
              default -> throw new AssertionError(action + " in a " + kind + " slot");
            };
        if (action.type() == Action.Type.CHI) {
          sub = action.base();
        }
      }
      default -> throw new AssertionError("Unknown decision kind " + kind);
    }
    return (rank * 100L + Math.max(sub, 0)) * 10 + selection;
  }

  private int ponIndex(int seat, int kind) {
    SeatView view = engine.view(seat);
    for (int index = 0; index < view.meldCount(seat); index++) {
      Meld meld = Meld.of(view.meld(seat, index));
      if (meld.type() == Meld.Type.PON && meld.base() == kind) {
        return index;
      }
    }
    throw new AssertionError("KAKAN without a pon of kind " + kind);
  }

  /**
   * 局末を照合する。局の結果は、局を終わらせた選択と状態に合う。点の移動と次の卓は、局の始まりの卓と結果を {@code Settlement.apply}
   * と {@code Progression.next} に渡した値と同じ。
   */
  private void checkRoundEnd() {
    RoundEnd end = engine.roundEnd();
    RoundOutcome outcome = end.outcome();
    FullView full = engine.full();
    SeatView table = full.seat(0);
    count(outcome.kind().name());
    if (outcome.kind() == EndKind.ABORTIVE) {
      count(outcome.abortive().name());
    }
    EventLog events = table.events();
    if (outcome.kind() == EndKind.RON && events.kind(events.size() - 1) == EventKind.NUKI) {
      count("RON_ON_NUKI");
    }
    checkOutcome(end, full);
    int riichi = 0;
    int tenpai = 0;
    int nagashi = 0;
    for (int seat = 0; seat < seats; seat++) {
      riichi |= table.riichi(seat) != RiichiState.NONE ? 1 << seat : 0;
      tenpai |= full.waits(seat) != 0 ? 1 << seat : 0;
      nagashi |= nagashi(table.events(), seat) ? 1 << seat : 0;
      load(full.seat(seat), seat);
      if (shape.size() == Variant.HAND) {
        Assert.assertEquals(end.waits(seat), full.waits(seat), "waits at the end of seat " + seat);
      }
    }
    Assert.assertEquals(outcome.riichiMask(), riichi);
    for (int i = 0; i < outcome.winnerCount(); i++) {
      Assert.assertEquals(outcome.base(i), end.score(i).base());
    }
    if (outcome.kind() == EndKind.EXHAUSTIVE) {
      Assert.assertEquals(outcome.tenpaiMask(), tenpai);
      Assert.assertEquals(outcome.nagashiMask(), nagashi);
    }
    int[] delta = new int[seats];
    Settlement.apply(variant, start, outcome, delta);
    TableState next = new TableState(variant);
    Progression.next(start, outcome, delta, next);
    int total = Variant.DEPOSIT * end.next().kyotaku();
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(end.delta(seat), delta[seat], "delta of seat " + seat);
      total += end.next().score(seat);
    }
    Fields.assertSameFields(next, end.next());
    Assert.assertEquals(total, seats * variant.startPoints(), "points and deposits after the round");
  }

  /**
   * 局を終わらせた選択から結果を組み立てて照合する。ロンを選んだ席があればロンで、判断のきっかけになったイベントの席が放銃者になり、
   * 和了者は放銃者から手番の近い順に並ぶ（三家和になる人数なら途中流局）。ツモ和了・九種九牌を選んだなら、その結果になる。ほかは、
   * 山の残りが 0 の荒牌か、打牌が通った後の途中流局。包の席はイベントから数え直す。
   */
  private void checkOutcome(RoundEnd end, FullView full) {
    RoundOutcome outcome = end.outcome();
    String text = outcome.toString();
    int[] winners = new int[seats - 1];
    int rons = 0;
    for (int offset = 1; offset < seats; offset++) {
      int seat = (actor + offset) % seats;
      for (int slot = 0; slot < pickedCount; slot++) {
        if (pickedSeats[slot] == seat && picked[slot] == Action.Type.RON_AGARI) {
          winners[rons++] = seat;
        }
      }
    }
    if (rons == AbortiveDraw.TRIPLE_RON_WINNERS && variant.aborts(AbortiveDraw.TRIPLE_RON)) {
      Assert.assertEquals(outcome.abortive(), AbortiveDraw.TRIPLE_RON, text);
    } else if (rons > 0) {
      Assert.assertEquals(outcome.kind(), EndKind.RON, text);
      Assert.assertEquals(outcome.loser(), actor, text);
      Assert.assertEquals(outcome.winnerCount(), rons, text);
      for (int i = 0; i < rons; i++) {
        Assert.assertEquals(outcome.winner(i), winners[i], text);
      }
    } else if (picked[0] == Action.Type.TSUMO_AGARI) {
      Assert.assertEquals(outcome.kind(), EndKind.TSUMO, text);
      Assert.assertEquals(outcome.winner(0), pickedSeats[0], text);
    } else if (picked[0] == Action.Type.KYUSHU_KYUHAI) {
      Assert.assertEquals(outcome.abortive(), AbortiveDraw.NINE_TERMINALS, text);
    } else if (outcome.kind() == EndKind.EXHAUSTIVE) {
      Assert.assertEquals(full.seat(0).liveLeft(), 0, text);
    } else {
      Assert.assertTrue(AFTER_DISCARD.contains(outcome.abortive()), text);
    }
    for (int i = 0; i < outcome.winnerCount(); i++) {
      int pao = pao(full.seat(0).events(), outcome.winner(i), outcome.loser(), end.score(i));
      Assert.assertEquals(outcome.pao(i), pao, text);
    }
  }

  /**
   * イベントから包の席を数え直す。大三元・大四喜の和了なら、和了者の三元牌・風牌の面子をイベントの順に数える。全ての種類がそろった
   * 面子について、鳴いた打牌を打った席が包になる。その面子が暗槓のとき、その席がロンの放銃者のとき、ほかの和了のときは
   * {@code RoundOutcome.NO_SEAT}。
   *
   * @param events 局のイベントの記録
   * @param winner 和了者
   * @param loser ロンの放銃者。ツモなら {@code RoundOutcome.NO_SEAT}
   * @param score 和了の点数
   * @return 包の席。包が無ければ {@code RoundOutcome.NO_SEAT}
   */
  private static int pao(EventLog events, int winner, int loser, WinScore score) {
    long group = score.has(ScoringYaku.DAISANGEN) ? DRAGONS : score.has(ScoringYaku.DAISUUSHII) ? WINDS : 0;
    int sets = 0;
    for (int event = 0; event < events.size(); event++) {
      if (events.kind(event) != EventKind.MELD || events.seat(event) != winner) {
        continue;
      }
      Meld meld = Meld.of(events.meld(event));
      if (Tiles.contains(group, meld.base()) && ++sets == Long.bitCount(group)) {
        int fed = meld.source() == Meld.RelativeSource.NONE ? RoundOutcome.NO_SEAT : events.seat(event - 1);
        return fed == loser ? RoundOutcome.NO_SEAT : fed;
      }
    }
    return RoundOutcome.NO_SEAT;
  }

  /** 河が空でなく、全てが么九牌で、どれも鳴かれていない（流し満貫の条件）。 */
  private static boolean nagashi(EventLog events, int seat) {
    for (int index = 0; index < events.riverSize(seat); index++) {
      int event = events.river(seat, index);
      if (events.calledBy(event) != EventLog.NONE || !Tiles.contains(Tiles.TERMINALS_AND_HONORS, events.tile(event))) {
        return false;
      }
    }
    return events.riverSize(seat) > 0;
  }

  /** 席の手（純手牌と面子の数）を作業用の手の形にコピーする。 */
  private void load(SeatView view, int seat) {
    shape.clear();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      for (int copy = 0; copy < view.count(kind); copy++) {
        shape.add(kind);
      }
    }
    for (int index = 0; index < view.meldCount(seat); index++) {
      shape.addMeld();
    }
  }

  private void count(String name) {
    coverage.merge(name, 1, Integer::sum);
  }
}
