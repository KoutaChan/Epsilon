package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.HypotheticalEnd;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.AllocationProbe;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Factory;
import org.testng.annotations.Test;

/**
 * 仮の局末と局の始まりの卓を、四人打ちと三人打ちで確かめる。自己対局の本当の局末で、局の結果をコピーして精算しても、結果の
 * フィールドから組み立て直して精算しても、点の動きと次局の卓はエンジンと同じになる（ダブロンと終局を含む）。局の始まりの卓は、
 * 配った局の卓と同じ。仮の局末は牌山を持たず、精算でメモリを確保しない。
 */
public class HypotheticalEndTest {

  private static final int GAMES = 300;

  private final Variant variant;
  private final int seats;
  private final RuleWorkspace ws;
  private final HypotheticalEnd hypothetical;
  private final Map<String, Integer> coverage = new TreeMap<>();

  @Factory(dataProvider = "variants")
  public HypotheticalEndTest(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    ws = new RuleWorkspace(variant);
    hypothetical = new HypotheticalEnd(variant);
  }

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test
  public void realRoundEndsSettleLikeTheEngine() {
    Engine engine = new Engine(variant);
    Assert.assertThrows(IllegalStateException.class, engine::handStart);
    engine.startGame(1, DrawOrder.WALL_ORDER);
    Assert.assertThrows(IllegalStateException.class, engine::handStart);
    RandomGames games = new RandomGames(engine, Boundary.GAME_END, 808, new GreedyChooser(variant));
    TableState expected = new TableState(variant);
    int finished = 0;
    while (finished < GAMES) {
      Boundary before = games.boundary();
      if (before == Boundary.ROUND_END) {
        expected.copyFrom(engine.roundEnd().next());
      } else if (before == Boundary.GAME_END) {
        expected.copyFrom(new TableState(variant));
      }
      Boundary reached = games.step();
      Assert.assertEquals(engine.handStart().toString(), expected.toString());
      if (reached != Boundary.DECISIONS) {
        assertSettlesLikeTheEngine(engine);
        count(reached == Boundary.GAME_END ? "GAME_END" : "ROUND_END");
      }
      if (reached == Boundary.GAME_END) {
        finished++;
      }
    }
    for (String seen : Set.of("TSUMO", "RON", "EXHAUSTIVE", "ABORTIVE", "GAME_END", "RIICHI")) {
      Assert.assertTrue(coverage.getOrDefault(seen, 0) > 0, variant + ": " + seen + " never happened: " + coverage);
    }
  }

  /** ダブロンの精算もエンジンと同じ。本場と供託は、放銃者から手番の近い最初の和了者だけが受け取る。 */
  @Test
  public void aDoubleRonSettlesLikeTheEngine() {
    EngineScene scene =
        variant == Variant.TENHOU_4P
            ? EngineScene.at(variant, 1, 2, 1, 25_000, 25_000, 24_000, 25_000)
                .hand(1, "5p147m147s369m123z")
                .hand(2, "123m456m789m46p11z")
                .hand(3, "234s567s678m34p55s")
                .hand(0, "19p258m1479s4567z")
                .draws("4z")
                .start()
            : EngineScene.at(variant, 0, 2, 1, 35_000, 35_000, 34_000)
                .hand(0, "19m258p147s369s67z")
                .hand(1, "123s456s789s46p11z")
                .hand(2, "234s678s678p34p55s")
                .draws("5p")
                .start();
    int loser = variant == Variant.TENHOU_4P ? 1 : 0;
    scene.play(loser + ":" + (variant == Variant.TENHOU_4P ? "DAHAI(5p)" : "DAHAI_TSUMOGIRI(5p)"));
    int first = (loser + 1) % seats;
    int second = (loser + 2) % seats;
    Assert.assertEquals(scene.play(first + ":RON_AGARI", second + ":RON_AGARI"), Boundary.ROUND_END);
    RoundOutcome outcome = scene.end().outcome();
    Assert.assertEquals(outcome.winnerCount(), 2);
    Assert.assertEquals(outcome.winner(0), first);
    Assert.assertEquals(outcome.winner(1), second);
    assertSettlesLikeTheEngine(scene.engine);
    TableState start = scene.engine.handStart();
    Assert.assertEquals(start.honba(), 2);
    Assert.assertEquals(start.kyotaku(), 1);
    TableState next =
        hypothetical.doubleRon(
            start, outcome.riichiMask(), loser, first, outcome.base(0), second, outcome.base(1));
    Assert.assertEquals(next.toString(), scene.end().next().toString());
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(hypothetical.delta(seat), scene.end().delta(seat), "seat " + seat);
    }
  }

  /** 仮の局末のフィールドは牌山の部品を持たない。精算は、局の始まりの卓のコピーと局の結果だけで決まる。 */
  @Test
  public void aHypotheticalEndNeverHoldsTheWall() {
    for (Class<?> type = HypotheticalEnd.class; type != Object.class; type = type.getSuperclass()) {
      for (Field field : type.getDeclaredFields()) {
        if (Modifier.isStatic(field.getModifiers())) {
          continue;
        }
        Class<?> held = field.getType().isArray() ? field.getType().getComponentType() : field.getType();
        Assert.assertTrue(
            held.isPrimitive() || held.getPackageName().equals("com.epsilon.mahjong.rules"),
            "HypotheticalEnd." + field.getName() + " holds " + held);
      }
    }
    RandomGames games = RandomGames.start(variant, 909, new GreedyChooser(variant));
    while (games.boundary() == Boundary.DECISIONS) {
      games.step();
    }
    TableState start = new TableState(variant);
    start.copyFrom(games.engine.handStart());
    RoundOutcome outcome = new RoundOutcome(variant);
    outcome.copyFrom(games.engine.roundEnd().outcome());
    String next = games.engine.roundEnd().next().toString();
    int[] delta = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      delta[seat] = games.engine.roundEnd().delta(seat);
    }
    while (games.step() == Boundary.DECISIONS) {}
    hypothetical.outcome().copyFrom(outcome);
    Assert.assertEquals(hypothetical.settle(start).toString(), next);
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(hypothetical.delta(seat), delta[seat], "seat " + seat);
    }
  }

  @Test
  public void settlingDoesNotAllocate() {
    RandomGames games = RandomGames.start(variant, 1010, new GreedyChooser(variant));
    while (games.boundary() == Boundary.DECISIONS) {
      games.step();
    }
    TableState start = games.engine.handStart();
    RoundOutcome outcome = games.engine.roundEnd().outcome();
    AllocationProbe.assertNoAllocation(
        20_000,
        100_000,
        rounds -> {
          long checksum = 0;
          for (int round = 0; round < rounds; round++) {
            hypothetical.outcome().copyFrom(outcome);
            checksum += hypothetical.settle(start).score(0);
            checksum += hypothetical.exhaustive(start, 0, round & 1, 0).score(1);
            checksum += hypothetical.tsumo(start, 0, round % seats, 2_000, RoundOutcome.NO_SEAT).score(0);
            checksum += hypothetical.ron(start, 0, 0, 1, 2_000, RoundOutcome.NO_SEAT).score(1);
          }
          return checksum;
        });
  }

  /** 局の結果をコピーした精算と、結果のフィールドから組み立て直した精算が、どちらもエンジンの点の動きと次局の卓に合う。 */
  private void assertSettlesLikeTheEngine(Engine engine) {
    RoundEnd end = engine.roundEnd();
    RoundOutcome outcome = end.outcome();
    TableState start = engine.handStart();
    hypothetical.outcome().copyFrom(outcome);
    assertLikeTheEngine(hypothetical.settle(start), end, "copied " + outcome);
    assertLikeTheEngine(rebuild(start, outcome), end, "rebuilt " + outcome);
    count(outcome.kind().name());
    if (outcome.riichiMask() != 0) {
      count("RIICHI");
    }
    if (outcome.kind() == EndKind.RON && outcome.winnerCount() == 2) {
      count("DOUBLE_RON");
    }
  }

  /** 局の結果のフィールドから、仮の局末のメソッド（ツモ・ロン・流局など）で同じ局末を作る。 */
  private TableState rebuild(TableState start, RoundOutcome outcome) {
    int riichi = outcome.riichiMask();
    return switch (outcome.kind()) {
      case TSUMO -> hypothetical.tsumo(start, riichi, outcome.winner(0), outcome.base(0), outcome.pao(0));
      case RON -> {
        if (outcome.winnerCount() == 1) {
          yield hypothetical.ron(
              start, riichi, outcome.loser(), outcome.winner(0), outcome.base(0), outcome.pao(0));
        }
        if (outcome.winnerCount() == 2 && outcome.pao(0) == RoundOutcome.NO_SEAT
            && outcome.pao(1) == RoundOutcome.NO_SEAT) {
          yield hypothetical.doubleRon(
              start, riichi, outcome.loser(), outcome.winner(0), outcome.base(0), outcome.winner(1), outcome.base(1));
        }
        hypothetical.outcome().setRon(outcome.base(0), outcome.winner(0), outcome.loser(), outcome.pao(0), riichi);
        for (int i = 1; i < outcome.winnerCount(); i++) {
          hypothetical.outcome().addRonWinner(outcome.base(i), outcome.winner(i), outcome.pao(i));
        }
        yield hypothetical.settle(start);
      }
      case EXHAUSTIVE -> hypothetical.exhaustive(start, riichi, outcome.tenpaiMask(), outcome.nagashiMask());
      case ABORTIVE -> hypothetical.abortive(start, riichi, outcome.abortive());
    };
  }

  private void assertLikeTheEngine(TableState next, RoundEnd end, String what) {
    Assert.assertSame(next, hypothetical.next());
    Assert.assertEquals(next.toString(), end.next().toString(), what);
    for (int seat = 0; seat < seats; seat++) {
      Assert.assertEquals(hypothetical.delta(seat), end.delta(seat), what + ", seat " + seat);
    }
  }

  private void count(String name) {
    coverage.merge(name, 1, Integer::sum);
  }
}
