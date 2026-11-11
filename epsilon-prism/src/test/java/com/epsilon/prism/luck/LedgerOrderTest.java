package com.epsilon.prism.luck;

import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * {@link LedgerTracker#observe} がイベントを読む順と、読んだ中身を確かめる。読んだイベントは {@link ChanceLog} のイベントと、種類・席・牌
 * まで 1 対 1 で一致する。各イベントの前のプールは、局の始めの枚数から前のイベントで出た牌を引いた値と一致する。手牌は、席のビューが持つ牌
 * から、前の判断の時点の後に引いた最後のツモを外した値と一致する。槓と抜きを必ず選ぶ打ち方で打ち、遅れた槓ドラの表示・連続の槓・嶺上の後の打牌を含める。
 */
public class LedgerOrderTest {

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P, 6000}, {Variant.TENHOU_3P, 4000}};
  }

  @Test(dataProvider = "variants", timeOut = 600_000)
  public void tracedEventsMatchTheChanceLog(Variant variant, int rounds) {
    TileSet set = variant.tiles();
    int seats = variant.seats();
    var tracker = new LedgerTracker(LedgerGames.RECORDING);
    List<LedgerTracker.Event> traced = new ArrayList<>();
    tracker.trace(traced);
    List<LedgerGames.Seat> targets = LedgerGames.seats(variant);
    int[] pool = new int[Tiles.KINDS];
    int[] counts = new int[4];
    boolean[] fresh = {true};
    int[] position = {0};
    LedgerGames.walk(
        variant,
        20261108L + seats,
        LedgerGames.CALLS,
        rounds,
        new LedgerGames.Visitor() {
          @Override
          public void decision(Engine engine, RuleWorkspace ws) {
            FullView full = engine.full();
            ChanceLog log = full.chance();
            if (fresh[0]) {
              // 最初の判断の時点より前のイベント（親の最初のツモ）は比べない
              tracker.observe(full, List.of(), ws);
              fresh[0] = false;
              position[0] = log.size();
              Assert.assertTrue(traced.isEmpty());
              return;
            }
            tracker.observe(full, targets, ws);
            Assert.assertEquals(traced.size(), log.size() - position[0], "events of the batch");
            int drawer = -1, drawn = -1;
            for (int i = position[0]; i < log.size(); i++)
              if (LedgerGames.draw(log.kind(i))) {
                drawer = log.seat(i);
                drawn = Tiles.kind(log.tile(i));
              }
            int[][] hands = new int[seats][Tiles.KINDS];
            for (int seat = 0; seat < seats; seat++) {
              SeatView view = full.seat(seat);
              for (long bits = set.kinds(); bits != 0; bits &= bits - 1) {
                int kind = Long.numberOfTrailingZeros(bits);
                for (int copy = 0; copy < Tiles.COPIES; copy++) {
                  int id = Tiles.physicalId(kind, copy);
                  if (set.contains(id) && view.holds(id)) hands[seat][kind]++;
                }
              }
              if (seat == drawer) hands[seat][drawn]--;
            }
            for (int k = 0; k < traced.size(); k++) {
              int index = position[0] + k;
              LedgerTracker.Event event = traced.get(k);
              String where = variant + " event " + index + " " + log.kind(index);
              Assert.assertEquals(event.kind(), log.kind(index), where);
              Assert.assertEquals(event.seat(), log.seat(index), where);
              Assert.assertEquals(event.tile(), log.tile(index), where);
              LedgerGames.poolBefore(log, index, variant, pool);
              Assert.assertEquals(event.pool(), pool, where + " pool");
              Assert.assertTrue(Arrays.deepEquals(event.hands(), hands), where + " hands");
              counts[event.kind().ordinal()]++;
            }
            traced.clear();
            position[0] = log.size();
          }

          @Override
          public void roundEnd(Engine engine) {
            tracker.reset();
            for (LedgerGames.Seat target : targets) Arrays.fill(target.ledger(), 0);
            fresh[0] = true;
          }
        });
    int compared = Arrays.stream(counts).sum();
    Assert.assertTrue(compared > 5 * rounds, variant + " compared " + compared);
    Assert.assertTrue(counts[ChanceKind.INDICATOR.ordinal()] >= 100, variant + " indicators " + Arrays.toString(counts));
    Assert.assertTrue(counts[ChanceKind.RINSHAN.ordinal()] >= 100, variant + " rinshan " + Arrays.toString(counts));
    Assert.assertEquals(counts[ChanceKind.URA.ordinal()], 0);
  }
}
