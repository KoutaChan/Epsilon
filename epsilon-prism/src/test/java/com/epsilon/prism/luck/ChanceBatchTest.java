package com.epsilon.prism.luck;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.ChanceLogs;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RandomGames;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 新しいエンジンの自己対局で、判断の時点ごとに偶然のイベントの束を読むテスト。束の並び、写しが親の読んだ位置から読み始めること、崩れた並びで
 * 例外を投げることを確かめる。
 */
public class ChanceBatchTest {

  private static final int ROUNDS = 600;

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void selfPlayBatchesAreOrderedAndContiguous(Variant variant) {
    RandomGames games = RandomGames.start(variant, 0x5eedL + variant.ordinal());
    ChanceBatch batch = new ChanceBatch();
    int rounds = 0;
    int read = 0;
    int reveals = 0;
    int rinshan = 0;
    while (rounds < ROUNDS) {
      switch (games.boundary()) {
        case DECISIONS -> {
          ChanceLog log = games.engine.full().chance();
          batch.read(log);
          Assert.assertEquals(batch.size(), log.size() - read, "The batch skips or repeats events");
          for (int k = 0; k < batch.reveals(); k++) {
            Assert.assertEquals(batch.reveal(k), read + k);
            Assert.assertEquals(log.kind(batch.reveal(k)), ChanceKind.INDICATOR);
          }
          if (batch.draw() >= 0) {
            Assert.assertEquals(batch.draw(), read + batch.reveals());
            ChanceKind kind = log.kind(batch.draw());
            Assert.assertTrue(kind == ChanceKind.DRAW || kind == ChanceKind.RINSHAN, kind.name());
            rinshan += kind == ChanceKind.RINSHAN ? 1 : 0;
          } else {
            Assert.assertEquals(batch.size(), batch.reveals());
          }
          reveals += batch.reveals();
          read = log.size();
        }
        case ROUND_END, GAME_END -> {
          batch.reset();
          read = 0;
          rounds++;
        }
      }
      games.step();
    }
    Assert.assertTrue(reveals > 0, "No kan indicator was seen");
    Assert.assertTrue(rinshan > 0, "No rinshan draw was seen");
  }

  @Test(dataProvider = "variants")
  public void copiesStartAtTheParentPosition(Variant variant) {
    RandomGames games = RandomGames.start(variant, 0xc0b1L + variant.ordinal());
    ChanceBatch parent = new ChanceBatch();
    int copies = 0;
    for (int step = 0; copies < 200; step++) {
      switch (games.boundary()) {
        case DECISIONS -> {
          parent.read(games.engine.full().chance());
          int position = games.engine.full().chance().size();
          if (step % 7 == 0) {
            ChanceBatch child = new ChanceBatch(parent);
            Assert.assertEquals(child.size(), 0);
            Assert.assertEquals(child.draw(), -1);
            Engine copy = games.engine.copy();
            RandomGames fork = new RandomGames(copy, Boundary.DECISIONS, step);
            if (fork.step() == Boundary.DECISIONS) {
              ChanceLog log = copy.full().chance();
              child.read(log);
              Assert.assertEquals(child.size(), log.size() - position);
              if (child.size() > 0) {
                Assert.assertEquals(child.reveals() > 0 ? child.reveal(0) : child.draw(), position);
              }
              copies++;
            }
          }
        }
        case ROUND_END, GAME_END -> parent.reset();
      }
      games.step();
    }
  }

  /**
   * 局末の決まり。判断の時点までの記録に裏ドラは無く、最後の判断の時点の後に残るのは、後めくりのドラ表示牌と、その後の裏ドラだけ。
   * 写したエンジンの記録は親と同じ（イベント、局の始めの枚数、誰にも見えない枚数）。
   */
  @Test(dataProvider = "variants")
  public void handEndLeavesLateIndicatorsThenUra(Variant variant) {
    RandomGames games = RandomGames.start(variant, 0xe4dL + variant.ordinal(), LedgerGames.chooser(variant, LedgerGames.KANS));
    int read = 0;
    int rounds = 0;
    int uras = 0;
    int late = 0;
    int copies = 0;
    for (int step = 0; rounds < ROUNDS; step++) {
      switch (games.boundary()) {
        case DECISIONS -> {
          ChanceLog log = games.engine.full().chance();
          for (int index = read; index < log.size(); index++) {
            Assert.assertNotEquals(log.kind(index), ChanceKind.URA);
          }
          read = log.size();
          if (step % 5 == 0) {
            ChanceLog copied = games.engine.copy().full().chance();
            Assert.assertEquals(copied.size(), log.size());
            for (int index = 0; index < log.size(); index++) {
              Assert.assertEquals(copied.kind(index), log.kind(index));
              Assert.assertEquals(copied.seat(index), log.seat(index));
              Assert.assertEquals(copied.tile(index), log.tile(index));
            }
            for (int kind = 0; kind < Tiles.KINDS; kind++) {
              Assert.assertEquals(copied.start(kind), log.start(kind));
              Assert.assertEquals(copied.unseen(kind), log.unseen(kind));
            }
            copies++;
          }
        }
        case ROUND_END, GAME_END -> {
          ChanceLog log = games.engine.full().chance();
          boolean ura = false;
          for (int index = read; index < log.size(); index++) {
            if (log.kind(index) == ChanceKind.URA) {
              ura = true;
              uras++;
            } else {
              Assert.assertEquals(log.kind(index), ChanceKind.INDICATOR, "event " + index + " after the last decision");
              Assert.assertFalse(ura, "an indicator after the ura indicators");
              late++;
            }
          }
          read = 0;
          rounds++;
        }
      }
      games.step();
    }
    Assert.assertTrue(uras > 0, "no ura indicator was seen (late indicators " + late + ")");
    Assert.assertTrue(copies > 0);
  }

  @Test
  public void malformedLogsThrow() {
    Variant variant = Variant.TENHOU_4P;
    expectThrow(ChanceLogs.of(variant, ChanceKind.DRAW, ChanceKind.INDICATOR));
    expectThrow(ChanceLogs.of(variant, ChanceKind.DRAW, ChanceKind.RINSHAN));
    expectThrow(ChanceLogs.of(variant, ChanceKind.RINSHAN, ChanceKind.DRAW));
    expectThrow(ChanceLogs.of(variant, ChanceKind.URA));
    expectThrow(ChanceLogs.of(variant, ChanceKind.INDICATOR, ChanceKind.URA));

    ChanceBatch batch = new ChanceBatch();
    batch.read(ChanceLogs.of(variant, ChanceKind.INDICATOR, ChanceKind.INDICATOR, ChanceKind.DRAW));
    Assert.assertThrows(IllegalStateException.class, () -> batch.read(ChanceLogs.of(variant, ChanceKind.DRAW)));

    ChanceBatch good = new ChanceBatch();
    good.read(ChanceLogs.of(variant, ChanceKind.INDICATOR, ChanceKind.INDICATOR, ChanceKind.RINSHAN));
    Assert.assertEquals(good.reveals(), 2);
    Assert.assertEquals(good.draw(), 2);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> good.reveal(2));
  }

  private static void expectThrow(ChanceLog log) {
    Assert.assertThrows(IllegalStateException.class, () -> new ChanceBatch().read(log));
  }
}
