package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.AllocationProbe;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Factory;
import org.testng.annotations.Test;

/**
 * 牌山のまだ読んでいない牌の並べ直しを、四人打ちと三人打ちで確かめる。自己対局の判断の時点でエンジンのコピーを並べ直し、元と
 * 比べる。
 *
 * <ul>
 *   <li>誰にも見えていない牌の種類ごとの枚数と、物理牌ごとに見えているかどうかは変わらない。
 *   <li>読んだ位置の牌は動かない。牌は、誰にも見えていない牌のある位置の間でだけ動く。
 *   <li>席別ツモ列のまだ使っていない項目は、元の牌があった位置に新しく来た牌に替わる。使った項目と列の長さは変わらない。
 *   <li>同じ種なら同じ牌山に、違う種なら違う牌山になる。同じ種で並べ直したコピーを同じ選び方で打つと、局の結果まで同じになる。
 *   <li>並べ直した後も、局末まで合法に打てる。
 *   <li>立直した席の和了の点数は、新しい牌山の裏ドラで採点し直してある。
 *   <li>同じ種なら、後の判断の時点の並びは、前の並びから既に読んだ牌を除いたものになる。
 *   <li>判断の時点を 1 つ固めて種を変えながら 2 万回並べ直すと、位置ごとの牌の種類は一様になる（χ² で検定する）。
 * </ul>
 */
public class ReshuffleUnseenTest {

  private static final int STEPS = 40_000;

  /** χ² 検定のしきい値。Wilson–Hilferty で正規分布に直した値の、両側 α = 1e-4 の点。 */
  private static final double CHI_SQUARE_Z = 3.8906;

  private final Variant variant;
  private final TileSet set;
  private final int seats;
  private final RuleWorkspace ws;

  @Factory(dataProvider = "variants")
  public ReshuffleUnseenTest(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    seats = variant.seats();
    ws = new RuleWorkspace(variant);
  }

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test
  public void reshufflingKeepsCountsReadTilesAndSeatColumns() {
    RandomGames games = RandomGames.start(variant, 101, new GreedyChooser(variant));
    int checks = 0;
    int streamed = 0;
    int differed = 0;
    int large = 0;
    int continued = 0;
    int riichiWins = 0;
    int uraChanged = 0;
    for (int step = 0; step < STEPS; step++) {
      Engine engine = games.engine;
      boolean riichiWin = games.boundary() == Boundary.DECISIONS && hasRiichiWin(engine);
      if (games.boundary() == Boundary.DECISIONS && (riichiWin || step % 8 == 0)) {
        long seed = step * 0x9E3779B97F4A7C15L;
        WallImage before = WallImage.of(engine);
        Engine copy = engine.copy();
        copy.reshuffleUnseen(seed, ws);
        WallImage after = WallImage.of(copy);
        assertReshuffled(before, after);
        checks++;
        streamed += before.streamed ? 1 : 0;

        Engine twin = engine.copy();
        twin.reshuffleUnseen(seed, ws);
        assertSameWall(WallImage.of(twin), after);
        if (before.hiddenCount() >= 10) {
          Engine other = engine.copy();
          other.reshuffleUnseen(seed + 1, ws);
          large++;
          differed += Arrays.equals(WallImage.of(other).tiles, after.tiles) ? 0 : 1;
        }
        if (riichiWin) {
          riichiWins++;
          uraChanged += assertRiichiWinsRescored(engine, copy);
        }
        if (riichiWin || checks % 4 == 0) {
          continued++;
          assertSameContinuation(copy, twin, seed);
        }
      }
      games.step();
    }
    Assert.assertTrue(checks > 2_000, checks + " checks");
    Assert.assertTrue(streamed > checks / 4 && streamed < checks * 3 / 4, streamed + " of " + checks + " streamed");
    Assert.assertTrue(differed >= large * 99 / 100, differed + " of " + large + " differ with another seed");
    Assert.assertTrue(continued > 500, continued + " continuations");
    Assert.assertTrue(riichiWins > 20, riichiWins + " riichi wins");
    Assert.assertTrue(uraChanged > 0, "no riichi win changed its ura in " + riichiWins);
  }

  /** 同じ種なら、後の判断の時点の並びは、前の並びから既に読んだ牌を除いたものになる。 */
  @Test
  public void aLaterReshuffleWithTheSameSeedKeepsTheEarlierOrder() {
    RandomGames games = RandomGames.start(variant, 202, new GreedyChooser(variant));
    long seed = 0x5DEECE66DL;
    int compared = 0;
    int shrunk = 0;
    for (int step = 0; step < STEPS && compared < 400; step++) {
      if (games.boundary() == Boundary.DECISIONS && step % 16 == 0) {
        int[] first = reshuffledOrder(games.engine, seed);
        int later = 1 + step % 12;
        while (later > 0 && games.step() == Boundary.DECISIONS) {
          later--;
        }
        if (later > 0) {
          continue;
        }
        FullView full = games.engine.full();
        int[] kept = Arrays.stream(first).filter(full::hidden).toArray();
        Assert.assertEquals(reshuffledOrder(games.engine, seed), kept, "step " + step);
        compared++;
        shrunk += kept.length < first.length ? 1 : 0;
      }
      games.step();
    }
    Assert.assertTrue(compared >= 100, compared + " comparisons");
    Assert.assertTrue(shrunk > compared / 2, shrunk + " of " + compared + " read a tile in between");
  }

  /**
   * 判断の時点を 1 つ固め、種を変えて 2 万回並べ直す。位置 × 種類の表で Pearson の χ² を求め、決まった牌の組を位置に配ることに
   * よる相関の分（U/(U − 1)、U は誰にも見えていない牌の数）で割る。これを自由度 (U − 1)(種類の数 − 1) の χ² 分布と比べる。
   */
  @Test
  public void positionsGetKindsUniformly() {
    RandomGames games = RandomGames.start(variant, 303, new GreedyChooser(variant));
    for (int step = 0; step < 15 || games.boundary() != Boundary.DECISIONS; step++) {
      games.step();
    }
    Engine copy = games.engine.copy();
    FullView full = copy.full();
    int[] tiles = (int[]) Fields.read(wall(copy), "tiles");
    int[] positions = new int[tiles.length];
    int unread = 0;
    for (int position = 0; position < tiles.length; position++) {
      if (full.hidden(tiles[position])) {
        positions[unread++] = position;
      }
    }
    int[][] counts = new int[unread][Tiles.KINDS];
    int runs = 20_000;
    for (int run = 0; run < runs; run++) {
      copy.reshuffleUnseen(0xC0FFEEL + run, ws);
      for (int i = 0; i < unread; i++) {
        counts[i][Tiles.kind(tiles[positions[i]])]++;
      }
    }
    double chiSquare = 0;
    int kinds = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      int hidden = full.unseen(kind);
      if (hidden == 0) {
        continue;
      }
      kinds++;
      double expected = (double) runs * hidden / unread;
      for (int i = 0; i < unread; i++) {
        double difference = counts[i][kind] - expected;
        chiSquare += difference * difference / expected;
      }
    }
    double scaled = chiSquare * (unread - 1) / unread;
    double freedom = (double) (unread - 1) * (kinds - 1);
    double z = (Math.cbrt(scaled / freedom) - (1 - 2 / (9 * freedom))) / Math.sqrt(2 / (9 * freedom));
    Assert.assertTrue(Math.abs(z) < CHI_SQUARE_Z, "chi-square " + scaled + " on " + freedom + " degrees, z " + z);
  }

  @Test
  public void reshufflingDoesNotAllocate() {
    RandomGames games = RandomGames.start(variant, 404, new GreedyChooser(variant));
    for (int step = 0; step < 20 || games.boundary() != Boundary.DECISIONS; step++) {
      games.step();
    }
    Engine copy = games.engine.copy();
    AllocationProbe.assertNoAllocation(
        20_000,
        100_000,
        rounds -> {
          long checksum = 0;
          for (int round = 0; round < rounds; round++) {
            copy.reshuffleUnseen(round, ws);
            checksum += copy.handStart().honba() + copy.full().unseen(round % Tiles.KINDS);
          }
          return checksum;
        });
  }

  @Test
  public void onlyDealtDecisionsOfAShuffledWallCanBeReshuffled() {
    Engine engine = new Engine(variant);
    Assert.assertThrows(IllegalStateException.class, () -> engine.reshuffleUnseen(1, ws));
    engine.startGame(1, DrawOrder.SEAT_STREAMS);
    Assert.assertThrows(IllegalStateException.class, () -> engine.reshuffleUnseen(1, ws));
    Assert.assertEquals(engine.advance(ws), Boundary.DECISIONS);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> engine.reshuffleUnseen(1, new RuleWorkspace(other(variant))));
    engine.reshuffleUnseen(1, ws);

    EngineScene scene = EngineScene.first(variant).start();
    Assert.assertThrows(IllegalStateException.class, () -> scene.engine.reshuffleUnseen(1, scene.ws));

    RandomGames games = new RandomGames(engine, Boundary.DECISIONS, 5, new GreedyChooser(variant));
    while (games.step() == Boundary.DECISIONS) {}
    Assert.assertThrows(IllegalStateException.class, () -> engine.reshuffleUnseen(1, ws));
  }

  private void assertReshuffled(WallImage before, WallImage after) {
    Assert.assertEquals(after.unseen, before.unseen, "unseen counts");
    Assert.assertEquals(after.hidden, before.hidden, "hidden tiles");
    int[] hiddenOfKind = new int[Tiles.KINDS];
    int[] places = new int[Tiles.PHYSICAL_IDS];
    for (int position = 0; position < before.tiles.length; position++) {
      int old = before.tiles[position];
      int tile = after.tiles[position];
      places[tile]++;
      if (before.hidden[old]) {
        Assert.assertTrue(before.hidden[tile], "position " + position + " got the seen tile " + tile);
        hiddenOfKind[Tiles.kind(tile)]++;
      } else {
        Assert.assertEquals(tile, old, "the read position " + position + " moved");
      }
    }
    for (int index = 0; index < set.size(); index++) {
      Assert.assertEquals(places[set.physicalId(index)], 1, "tile " + set.physicalId(index) + " on the wall");
    }
    Assert.assertEquals(hiddenOfKind, before.unseen, "hidden tiles per kind on the wall");

    Assert.assertEquals(after.streamed, before.streamed);
    Assert.assertEquals(after.streamSize, before.streamSize);
    Assert.assertEquals(after.streamCursor, before.streamCursor);
    if (!before.streamed) {
      return;
    }
    for (int position = 0; position < after.tiles.length; position++) {
      Assert.assertEquals(after.positionOf[after.tiles[position]], position, "position of " + after.tiles[position]);
    }
    for (int seat = 0; seat < seats; seat++) {
      int base = seat * before.streamCapacity;
      for (int index = 0; index < before.streamSize[seat]; index++) {
        int old = before.stream[base + index];
        int expected =
            index >= before.streamCursor[seat] && before.hidden[old] ? after.tiles[before.positionOf[old]] : old;
        Assert.assertEquals(after.stream[base + index], expected, "seat " + seat + " column item " + index);
      }
    }
  }

  private static void assertSameWall(WallImage actual, WallImage expected) {
    Assert.assertEquals(actual.tiles, expected.tiles);
    Assert.assertEquals(actual.stream, expected.stream);
    Assert.assertEquals(actual.positionOf, expected.positionOf);
  }

  /**
   * コピーでの立直の和了の点数が、新しい牌山で採点し直した点数と同じで、元の点数とは裏ドラだけが違うことを確かめる。
   *
   * @return 裏ドラの数が元と違った和了の数
   */
  private int assertRiichiWinsRescored(Engine original, Engine copy) {
    Object round = Fields.read(copy, "round");
    PublicState state = (PublicState) Fields.read(round, "state");
    Hand[] hands = (Hand[]) Fields.read(round, "hands");
    Wall wall = wall(copy);
    Decisions decisions = (Decisions) Fields.read(round, "decisions");
    Decisions before = (Decisions) Fields.read(Fields.read(original, "round"), "decisions");
    int changed = 0;
    for (int slot = 0; slot < decisions.size(); slot++) {
      int seat = decisions.seat(slot);
      if (!decisions.hasWin(slot) || state.riichi(seat) == RiichiState.NONE) {
        continue;
      }
      WinScore expected = new WinScore();
      TurnEvent last = state.last();
      boolean won =
          decisions.kind(slot) == EngineDecisionKind.TURN
              ? WinLegality.tsumo(state, hands[seat], wall, ws, expected)
              : WinLegality.ron(
                  state, hands[seat], seat, last.tile(), last.red(), Window.ronSource(state), wall, ws, expected);
      Assert.assertTrue(won);
      WinScore score = decisions.win(slot);
      WinScore old = before.win(slot);
      Assert.assertEquals(score.toString(), expected.toString(), "seat " + seat);
      Assert.assertEquals(score.yaku(), old.yaku());
      Assert.assertEquals(score.fu(), old.fu());
      Assert.assertEquals(score.dora(), old.dora());
      Assert.assertEquals(score.aka(), old.aka());
      Assert.assertEquals(score.nuki(), old.nuki());
      Assert.assertEquals(score.yakuman(), old.yakuman());
      if (score.yakuman() == 0) {
        Assert.assertEquals(score.han() - old.han(), score.ura() - old.ura(), score + " after " + old);
      }
      changed += score.ura() != old.ura() ? 1 : 0;
    }
    return changed;
  }

  /**
   * 同じ種で並べ直した 2 つのコピーを、同じ種の選び方で局末まで打つ。どの判断の時点でも牌の数が合い、偶然のイベントと局の結果も
   * 同じになる。
   */
  private void assertSameContinuation(Engine copy, Engine twin, long seed) {
    RandomGames first = new RandomGames(copy, Boundary.DECISIONS, seed, new GreedyChooser(variant));
    RandomGames second = new RandomGames(twin, Boundary.DECISIONS, seed, new GreedyChooser(variant));
    Boundary boundary = Boundary.DECISIONS;
    while (boundary == Boundary.DECISIONS) {
      assertTilesAccounted(copy);
      boundary = first.step();
      Assert.assertEquals(second.step(), boundary);
    }
    ChanceLog left = copy.full().chance();
    ChanceLog right = twin.full().chance();
    Assert.assertEquals(right.size(), left.size());
    for (int index = 0; index < left.size(); index++) {
      Assert.assertEquals(right.tile(index), left.tile(index), "chance event " + index);
      Assert.assertEquals(right.kind(index), left.kind(index), "chance event " + index);
    }
    Assert.assertEquals(twin.roundEnd().outcome().toString(), copy.roundEnd().outcome().toString());
    Assert.assertEquals(twin.roundEnd().next().toString(), copy.roundEnd().next().toString());
    int total = copy.roundEnd().next().kyotaku() * Variant.DEPOSIT;
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(twin.roundEnd().delta(seat), copy.roundEnd().delta(seat));
      total += copy.roundEnd().next().score(seat);
    }
    Assert.assertEquals(total, seats * variant.startPoints(), "points and deposits");
  }

  /**
   * どの物理牌も、手にあるなら 1 人の手にだけある。手にある牌は「誰にも見えていない牌」に入らない。見えていない牌を種類ごとに数えると、
   * 見えていない枚数と合う。牌山にはルールの牌がちょうど 1 枚ずつ並ぶ。
   */
  private void assertTilesAccounted(Engine engine) {
    FullView full = engine.full();
    int[] hiddenOfKind = new int[Tiles.KINDS];
    for (int index = 0; index < set.size(); index++) {
      int tile = set.physicalId(index);
      int holders = 0;
      for (int seat = 0; seat < seats; seat++) {
        holders += full.seat(seat).holds(tile) ? 1 : 0;
      }
      Assert.assertTrue(holders <= 1, "tile " + tile + " is in " + holders + " hands");
      if (full.hidden(tile)) {
        Assert.assertEquals(holders, 0, "the hidden tile " + tile + " is in a hand");
        hiddenOfKind[Tiles.kind(tile)]++;
      }
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      Assert.assertEquals(hiddenOfKind[kind], full.unseen(kind), "kind " + kind);
    }
    int[] places = new int[Tiles.PHYSICAL_IDS];
    for (int tile : (int[]) Fields.read(wall(engine), "tiles")) {
      places[tile]++;
    }
    for (int index = 0; index < set.size(); index++) {
      Assert.assertEquals(places[set.physicalId(index)], 1, "tile " + set.physicalId(index) + " on the wall");
    }
  }

  /** エンジンのコピーを種で並べ直し、誰にも見えていない牌を牌山の位置の順に並べて返す。 */
  private int[] reshuffledOrder(Engine engine, long seed) {
    Engine copy = engine.copy();
    copy.reshuffleUnseen(seed, ws);
    FullView full = copy.full();
    return Arrays.stream((int[]) Fields.read(wall(copy), "tiles")).filter(full::hidden).toArray();
  }

  private boolean hasRiichiWin(Engine engine) {
    Decisions decisions = engine.decisions();
    for (int slot = 0; slot < decisions.size(); slot++) {
      if (decisions.hasWin(slot) && engine.view(0).riichi(decisions.seat(slot)) != RiichiState.NONE) {
        return true;
      }
    }
    return false;
  }

  private static Wall wall(Engine engine) {
    return (Wall) Fields.read(Fields.read(engine, "round"), "wall");
  }

  private static Variant other(Variant variant) {
    return variant == Variant.TENHOU_4P ? Variant.TENHOU_3P : Variant.TENHOU_4P;
  }

  /** 並べ直しの前後で比べる、牌山の状態のコピー（牌の並び、席別ツモ列、物理牌ごとに誰にも見えていないか、種類ごとの見えていない枚数）。 */
  private record WallImage(
      int[] tiles,
      int[] positionOf,
      int[] stream,
      int[] streamSize,
      int[] streamCursor,
      int streamCapacity,
      boolean streamed,
      boolean[] hidden,
      int[] unseen) {

    static WallImage of(Engine engine) {
      Wall wall = wall(engine);
      FullView full = engine.full();
      TileSet set = full.variant().tiles();
      boolean[] hidden = new boolean[Tiles.PHYSICAL_IDS];
      for (int index = 0; index < set.size(); index++) {
        hidden[set.physicalId(index)] = full.hidden(set.physicalId(index));
      }
      int[] unseen = new int[Tiles.KINDS];
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        unseen[kind] = full.unseen(kind);
      }
      return new WallImage(
          ((int[]) Fields.read(wall, "tiles")).clone(),
          ((int[]) Fields.read(wall, "positionOf")).clone(),
          ((int[]) Fields.read(wall, "stream")).clone(),
          ((int[]) Fields.read(wall, "streamSize")).clone(),
          ((int[]) Fields.read(wall, "streamCursor")).clone(),
          (Integer) Fields.read(wall, "streamCapacity"),
          (Boolean) Fields.read(wall, "streamed"),
          hidden,
          unseen);
    }

    int hiddenCount() {
      int count = 0;
      for (int value : unseen) {
        count += value;
      }
      return count;
    }
  }
}
