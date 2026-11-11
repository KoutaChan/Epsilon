package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * エンジンを確かめる。見るのは、呼び出しの検査（段階に合わない呼び出しと不正な選択を拒む）、牌譜の 1 局、決定性、コピーの独立。
 * 決定性とコピーは四人打ちと三人打ちの両方で確かめる。
 */
public class EngineTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test
  public void eachCallNeedsItsStageAndValidChoices() {
    Engine engine = new Engine(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    Assert.assertThrows(IllegalStateException.class, () -> engine.advance(ws));
    Assert.assertThrows(IllegalStateException.class, () -> engine.commit(new int[VARIANT.seats() - 1], ws));
    Assert.assertThrows(IllegalStateException.class, engine::decisions);
    Assert.assertThrows(IllegalStateException.class, engine::roundEnd);
    Assert.assertThrows(IllegalStateException.class, engine::copy);
    engine.startGame(5, DrawOrder.WALL_ORDER);
    Assert.assertThrows(IllegalArgumentException.class, () -> engine.advance(new RuleWorkspace(Variant.TENHOU_3P)));
    Assert.assertEquals(engine.advance(ws), Boundary.DECISIONS);
    Assert.assertThrows(IllegalStateException.class, () -> engine.advance(ws));
    Decisions decisions = engine.decisions();
    Assert.assertEquals(decisions.size(), 1);
    Assert.assertEquals(decisions.kind(0), EngineDecisionKind.TURN);
    Assert.assertEquals(decisions.seat(0), 0);
    Assert.assertEquals(decisions.id(0), 1L);
    Assert.assertThrows(IllegalArgumentException.class, () -> engine.commit(new int[0], ws));
    Assert.assertThrows(IllegalArgumentException.class, () -> engine.commit(new int[] {-1}, ws));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> engine.commit(new int[] {decisions.legalCount(0)}, ws));
    Assert.assertEquals(engine.decisions().id(0), 1L, "a rejected commit changes nothing");
  }

  /** 牌譜の局は 1 局だけで、局末の後に次の局は配れない。 */
  @Test
  public void aRecordedRoundEndsAtItsRoundEnd() {
    EngineScene scene = EngineScene.first(VARIANT).hand(0, "19m19p19s1234z234m").draws("5z").start();
    Assert.assertEquals(scene.play("0:KYUSHU_KYUHAI"), Boundary.ROUND_END);
    Assert.assertThrows(IllegalStateException.class, () -> scene.engine.advance(scene.ws));
  }

  /** 半荘を最初から始めると、東 1 局の卓から牌山の epoch 1 で始めたときと同じ半荘になる。 */
  @Test(dataProvider = "variants")
  public void startGameIsStartAtTheFirstTableWithTheFirstEpoch(Variant variant) {
    RuleWorkspace ws = new RuleWorkspace(variant);
    Engine fromGame = new Engine(variant);
    fromGame.startGame(41, DrawOrder.SEAT_STREAMS);
    Engine fromTable = new Engine(variant);
    fromTable.startAt(RoundStart.first(variant), 41, 1, DrawOrder.SEAT_STREAMS);
    RandomGames game = new RandomGames(fromGame, fromGame.advance(ws), 9);
    RandomGames table = new RandomGames(fromTable, fromTable.advance(ws), 9);
    while (game.boundary() != Boundary.GAME_END) {
      Assert.assertEquals(table.step(), game.step());
      if (game.boundary() != Boundary.DECISIONS) {
        Fields.assertSameFields(fromGame, fromTable);
      }
    }
  }

  /** 乱数の種が同じで同じ手を選べば、全ての局末で全てのフィールド（イベント・偶然のイベント・点を含む）が同じになる。 */
  @Test(dataProvider = "variants")
  public void theSameSeedReplaysTheSameGames(Variant variant) {
    RandomGames first = RandomGames.start(variant, 77);
    RandomGames second = RandomGames.start(variant, 77);
    int games = 0;
    while (games < 3) {
      Assert.assertEquals(second.step(), first.step());
      if (first.boundary() != Boundary.DECISIONS) {
        Fields.assertSameFields(first.engine, second.engine);
        games += first.boundary() == Boundary.GAME_END ? 1 : 0;
      }
    }
  }

  /** 手の選び方を渡した RandomGames は、判断する席ごとに 1 度だけ選び方を呼ぶ。乱数の種と選び方が同じなら、同じ半荘になる。 */
  @Test(dataProvider = "variants")
  public void aChooserPicksEveryMoveOfTheWalker(Variant variant) {
    int[] asked = new int[1];
    RandomGames.Chooser last =
        (engine, slot, random) -> {
          asked[0]++;
          return engine.decisions().legalCount(slot) - 1;
        };
    RandomGames games = RandomGames.start(variant, 21, last);
    int slots = 0;
    for (int step = 0; step < 2_000; step++) {
      if (games.boundary() == Boundary.DECISIONS) {
        slots += games.engine.decisions().size();
      }
      games.step();
    }
    Assert.assertTrue(slots > 0);
    Assert.assertEquals(asked[0], slots);

    RandomGames.Chooser mixed =
        (engine, slot, random) -> random.nextBoolean() ? 0 : random.nextInt(engine.decisions().legalCount(slot));
    RandomGames first = RandomGames.start(variant, 5, mixed);
    RandomGames second = RandomGames.start(variant, 5, mixed);
    for (int step = 0; step < 5_000; step++) {
      Assert.assertEquals(second.step(), first.step());
      if (first.boundary() != Boundary.DECISIONS) {
        Fields.assertSameFields(first.engine, second.engine);
      }
    }
  }

  /**
   * 判断の時点で取ったコピーは、全てのフィールドが元と等しく、元と何も共有しない。同じ手を選べば元と同じ所へ着き、コピーを進めても
   * 元は変わらない。
   */
  @Test(dataProvider = "variants")
  public void aCopyContinuesLikeTheOriginalAndLeavesItAlone(Variant variant) {
    Random random = new Random(64);
    RandomGames original = RandomGames.start(variant, 3);
    for (int trial = 0; trial < 30; trial++) {
      int steps = random.nextInt(150);
      for (int step = 0; step < steps || original.boundary() != Boundary.DECISIONS; step++) {
        original.step();
      }
      Engine copy = original.engine.copy();
      Fields.assertSameFields(original.engine, copy);
      Engine snapshot = original.engine.copy();
      long rest = random.nextLong();
      RandomGames branch = new RandomGames(copy, Boundary.DECISIONS, rest);
      for (int step = 0; step < 300; step++) {
        branch.step();
      }
      Fields.assertSameFields(snapshot, original.engine);
      original = new RandomGames(original.engine, Boundary.DECISIONS, rest);
      for (int step = 0; step < 300; step++) {
        original.step();
      }
      Fields.assertSameFields(original.engine, copy);
    }
  }
}
