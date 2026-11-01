package com.epsilon.prism.collect;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.WinClaim;
import com.epsilon.util.SeedMixer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 運の精算の法則（{@link EventLedger}）: 引く牌は全員から見えない牌のプール n_t / N の上で一様（並べ替えた牌山で数える・本物の牌山で PIT）で、
 * 局面を固定して牌山だけを並べ替えると帳簿の増分の平均は局面ごとに 0（c が F_k の外を読む漏れを捕まえる。局末の裏ドラの項も席ごとに）。
 */
public class EventLedgerLawTest {
  private static final int TILES = Tile.NUM_TILE_TYPES;

  /** 境界の写し: engine の複製の元、境界の判断、選んだ手、選びの乱数の種、境界のプール。 */
  private record Boundary(GameEngine engine, List<EngineDecisionPoint> points, EngineSelectionBuffer selections, long choiceSeed, int[] pool, boolean ankan) {}

  /** TURN の境界でツモの直後か。 */
  private static boolean freshDraw(GameState state, List<EngineDecisionPoint> points) {
    return points.size() == 1
        && points.getFirst().kind() == EngineDecisionKind.TURN
        && state.turnEvent() instanceof TurnEvent.Draw draw
        && draw.player() == points.getFirst().player()
        && draw.tileType() >= 0;
  }

  /**
   * 境界から選んだ手を commit し、同じ種の打ち方で、次の TURN の境界でツモの直後まで進める（並べ替えた牌山でも手は同じ）。局が終われば null。
   */
  private static GameState nextDraw(GameEngine fork, EngineSelectionBuffer selections, long choiceSeed, LedgerGames.Style style) {
    var shapes = new HandShapeAnalyzer();
    var random = new SplittableRandom(choiceSeed);
    GameStepResult step = fork.commitDecisions(selections);
    for (int guard = 0; guard < 64; guard++) {
      if (!(step instanceof GameStepResult.AwaitingDecisions awaiting)) return null;
      List<EngineDecisionPoint> points = awaiting.decisions();
      if (freshDraw(fork.getState(), points)) return fork.getState();
      step = fork.commitDecisions(LedgerGames.select(fork.getState(), points, style, random, shapes, null));
    }
    throw new IllegalStateException("No draw within 64 boundaries");
  }

  /**
   * 並べ替えた牌山のツモ: 自己対局の境界をいくつか（暗槓を選ぶ境界を含む）選び、各境界で複製して全員から見えない牌だけ並べ替え
   * （{@link GameState#reshuffleUnseen}）、同じ手を commit した次のツモの種類を数える。期待は境界のプール n_t / N（暗槓は表示牌の分を周辺化した
   * 嶺上の牌の分布）。王牌・裏・未公開の表示牌・嶺上・赤の数え方と、プールの式を確かめる。
   */
  @Test(timeOut = 600_000)
  public void poolMatchesTheUnseenSet() {
    List<Boundary> boundaries = new ArrayList<>();
    var shapes = new HandShapeAnalyzer();
    int plain = 0, ankans = 0;
    for (long seed = 1; seed < 400 && (plain < 9 || ankans < 2); seed++) {
      var engine = new GameEngine(SeedMixer.mix64(seed * 7919));
      engine.getState().enableSeatStreamDraws();
      GameStepResult step = engine.stepHanchan();
      int turn = 0;
      while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
        List<EngineDecisionPoint> points = awaiting.decisions();
        GameState state = engine.getState();
        long choiceSeed = SeedMixer.mix64(seed ^ (turn * 31L));
        if (freshDraw(state, points)) {
          turn++;
          boolean ankan = false;
          for (Action action : points.getFirst().legalActions()) ankan |= action.type() == Action.Type.ANKAN;
          boolean wanted = ankan ? ankans < 2 : plain < 9 && turn % 11 == 5;
          if (wanted) {
            int[] pool = new int[TILES];
            EventLedger.pool(state, pool);
            var selections = LedgerGames.select(state, points, LedgerGames.KANS, new SplittableRandom(choiceSeed), shapes, null);
            GameEngine probe = engine.forkAtDecisionBoundary();
            int indicators = state.doraState().indicatorCount();
            GameState next = nextDraw(probe, selections, choiceSeed, LedgerGames.KANS);
            // 表示が増えた境界は暗槓（嶺上の前に表示）だけを使う（遅れた表示は境界のプールを変える）
            if (next != null && (next.doraState().indicatorCount() == indicators || ankan)) {
              boundaries.add(new Boundary(engine.forkAtDecisionBoundary(), points, selections, choiceSeed, pool, ankan));
              if (ankan) ankans++;
              else plain++;
            }
          }
        }
        step = engine.commitDecisions(LedgerGames.select(state, points, LedgerGames.KANS, new SplittableRandom(choiceSeed), shapes, null));
        if (step instanceof GameStepResult.RoundSettled || step instanceof GameStepResult.HanchanEnded) break;
      }
    }
    Assert.assertTrue(boundaries.size() >= 10, "boundaries " + boundaries.size());
    Assert.assertTrue(ankans >= 1, "no boundary with a concealed kan");
    int replicates = 4000;
    for (int b = 0; b < boundaries.size(); b++) {
      Boundary boundary = boundaries.get(b);
      long[] observed = new long[TILES];
      for (int r = 0; r < replicates; r++) {
        GameEngine fork = boundary.engine().forkAtDecisionBoundary();
        fork.getState().reshuffleUnseen(SeedMixer.mix64((b + 1L) * 1_000_003L + r));
        GameState next = nextDraw(fork, boundary.selections(), boundary.choiceSeed(), LedgerGames.KANS);
        Assert.assertNotNull(next, "replicate " + r + " of boundary " + b + " ended the hand");
        observed[((TurnEvent.Draw) next.turnEvent()).tileType()]++;
      }
      int total = 0;
      for (int count : boundary.pool()) total += count;
      double[] expected = new double[TILES];
      for (int t = 0; t < TILES; t++) expected[t] = replicates * (double) boundary.pool()[t] / total;
      double statistic = LedgerGames.chiSquare(observed, expected);
      double p = LedgerGames.chiSquareUpper(statistic, LedgerGames.degrees(expected));
      Assert.assertTrue(p > 1e-4, "boundary " + b + " (ankan " + boundary.ankan() + ") chi2=" + statistic + " p=" + p);
    }
  }

  /** PIT の区分の数。 */
  private static final int BINS = 20;

  /** 乱択化した PIT の 20 区分の度数（{@link #add}）と件数。 */
  private static final class Pit {
    final long[] bins = new long[BINS];
    long count;

    /** 種類 tile を引いた（プール pool、独立な一様 v）。u = (Σ_{t<tile} n_t + v·n_tile) / N。 */
    void add(int[] pool, int tile, double v) {
      int below = 0, total = 0;
      for (int t = 0; t < TILES; t++) {
        if (t < tile) below += pool[t];
        total += pool[t];
      }
      Assert.assertTrue(pool[tile] > 0, "drew tile " + tile + " outside the pool");
      double u = (below + v * pool[tile]) / total;
      bins[Math.min(BINS - 1, (int) (u * BINS))]++;
      count++;
    }

    double p() {
      double[] expected = new double[BINS];
      Arrays.fill(expected, count / (double) BINS);
      return LedgerGames.chiSquareUpper(LedgerGames.chiSquare(bins, expected), BINS - 1);
    }
  }

  /**
   * 本物の牌山（並べ替えない）: 局を多く打ち、TURN の境界でツモの直後ごとに、ツモの前のプール（今のプールに引いた牌を戻したもの）で乱択化した PIT を
   * 20 区分のカイ二乗で検定する（WASO と牌山の並べ替えの交換可能性）。部分集合も別に検定する: 席の列が尽きた後のツモ（席ごとの生きた山のツモの
   * 回数が列の長さを超えたツモ）、嶺上、槓ドラの表示（表示の前のプール。記録処理のフックで読む）。
   */
  @Test(timeOut = 600_000)
  public void drawsAreUniformOverThePool() {
    var shapes = new HandShapeAnalyzer();
    var pitRandom = new SplittableRandom(20261028L);
    Pit all = new Pit(), exhausted = new Pit(), rinshan = new Pit(), reveals = new Pit();
    int[] pool = new int[TILES];
    for (long seed = 1; seed <= 10_000; seed++) {
      var revealPool = new int[TILES];
      var recorder =
          new LedgerGames.SilentRecorder() {
            @Override
            public void beforeDoraReveal(GameState state) {
              EventLedger.pool(state, revealPool);
            }

            @Override
            public void recordNewDora(int physicalTileId) {
              reveals.add(revealPool, Tile.typeOf(physicalTileId), pitRandom.nextDouble());
            }
          };
      var engine = new GameEngine(SeedMixer.mix64(seed * 104_729), recorder);
      engine.getState().enableSeatStreamDraws();
      var random = new SplittableRandom(seed * 13);
      LedgerGames.Style style = seed % 2 == 0 ? LedgerGames.CALLS : LedgerGames.KANS;
      int[] liveDraws = new int[GameState.NUM_PLAYERS];
      GameStepResult step = engine.stepHanchan();
      while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
        List<EngineDecisionPoint> points = awaiting.decisions();
        GameState state = engine.getState();
        if (freshDraw(state, points)) {
          var draw = (TurnEvent.Draw) state.turnEvent();
          EventLedger.pool(state, pool);
          pool[draw.tileType()]++;
          double v = pitRandom.nextDouble();
          all.add(pool, draw.tileType(), v);
          if (draw.isRinshanDraw()) {
            rinshan.add(pool, draw.tileType(), v);
          } else {
            int seat = draw.player();
            liveDraws[seat]++;
            int offset = Math.floorMod(seat - state.getOya(), GameState.NUM_PLAYERS);
            if (liveDraws[seat] > (70 + 3 - offset) / 4) exhausted.add(pool, draw.tileType(), v);
          }
        }
        step = engine.commitDecisions(LedgerGames.select(state, points, style, random, shapes, null));
        if (step instanceof GameStepResult.RoundSettled || step instanceof GameStepResult.HanchanEnded) break;
      }
    }
    Assert.assertTrue(all.count > 400_000, "draws " + all.count);
    Assert.assertTrue(exhausted.count >= 1, "no draw after an exhausted seat stream");
    Assert.assertTrue(rinshan.count >= 100, "rinshan draws " + rinshan.count);
    Assert.assertTrue(reveals.count >= 100, "dora reveals " + reveals.count);
    Assert.assertTrue(all.p() > 1e-4, "all draws p=" + all.p());
    Assert.assertTrue(rinshan.p() > 1e-4, "rinshan draws p=" + rinshan.p());
    Assert.assertTrue(reveals.p() > 1e-4, "dora reveals p=" + reveals.p());
    if (exhausted.count >= 200) Assert.assertTrue(exhausted.p() > 1e-4, "exhausted streams p=" + exhausted.p() + " n=" + exhausted.count);
  }

  /**
   * 条件付きの平均 0: 境界をいくつか（聴牌の他家・立直の席・遅れた表示・暗槓を選べる境界を含む）固定し、各境界で牌山だけを並べ替えた分岐を
   * 8000 本作り、同じ手を commit して次の出来事（ツモ・嶺上・表示）までの帳簿の増分を席と欄ごとに平均する。並べ替えは F_k を変えずに見えない牌
   * だけを引き直すので、c がツモの後の手・裏ドラ・引いた牌の赤など F_k の外を読んでいれば、その境界の平均がずれる。境界ごとに |z| &lt; 4.5。
   */
  @Test(timeOut = 600_000)
  public void conditionalMeanIsZeroUnderReshuffle() {
    var shapes = new HandShapeAnalyzer();
    List<Boundary> boundaries = new ArrayList<>();
    int[] wanted = {8, 8, 6, 6};
    int[] found = new int[4];
    for (long seed = 1; seed < 4000 && boundaries.size() < 28; seed++) {
      var engine = new GameEngine(SeedMixer.mix64(seed * 15_485_863L));
      engine.getState().enableSeatStreamDraws();
      var random = new SplittableRandom(seed * 17);
      GameStepResult step = engine.stepHanchan();
      int turn = 0;
      while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
        List<EngineDecisionPoint> points = awaiting.decisions();
        GameState state = engine.getState();
        long choiceSeed = SeedMixer.mix64(seed * 1_000_003L + turn++);
        if (freshDraw(state, points)) {
          int category = category(state, points.getFirst(), shapes);
          if (category >= 0 && found[category] < wanted[category]) {
            var selections = LedgerGames.select(state, points, LedgerGames.KANS, new SplittableRandom(choiceSeed), shapes, null);
            int[] pool = new int[TILES];
            EventLedger.pool(state, pool);
            boundaries.add(new Boundary(engine.forkAtDecisionBoundary(), points, selections, choiceSeed, pool, category == 3));
            found[category]++;
          }
        }
        step = engine.commitDecisions(LedgerGames.select(state, points, LedgerGames.KANS, random, shapes, null));
        if (step instanceof GameStepResult.RoundSettled || step instanceof GameStepResult.HanchanEnded) break;
      }
    }
    Assert.assertTrue(boundaries.size() >= 20, "boundaries " + boundaries.size() + " " + Arrays.toString(found));
    int replicates = 8000, tested = 0;
    for (int b = 0; b < boundaries.size(); b++) {
      Boundary boundary = boundaries.get(b);
      double[][] sum = new double[GameState.NUM_PLAYERS][EventLedger.FEATURES];
      double[][] square = new double[GameState.NUM_PLAYERS][EventLedger.FEATURES];
      int counted = 0;
      for (int r = 0; r < replicates; r++) {
        GameEngine fork = boundary.engine().forkAtDecisionBoundary();
        fork.getState().reshuffleUnseen(SeedMixer.mix64((b + 7L) * 998_244_353L + r));
        var tracker = new EventLedger.Tracker(new EventLedger.Tracker(), boundary.engine().getState());
        List<EventLedger.Event> events = new ArrayList<>();
        tracker.trace(events);
        List<Slot> slots = LedgerGames.slots();
        var random = new SplittableRandom(boundary.choiceSeed());
        GameStepResult step = fork.commitDecisions(boundary.selections());
        while (events.isEmpty() && step instanceof GameStepResult.AwaitingDecisions awaiting) {
          tracker.observe(fork.getState(), awaiting.decisions(), slots, null);
          if (!events.isEmpty()) break;
          step = fork.commitDecisions(LedgerGames.select(fork.getState(), awaiting.decisions(), LedgerGames.KANS, random, shapes, null));
        }
        if (events.isEmpty()) continue;
        counted++;
        for (Slot slot : slots)
          for (int f = 0; f < EventLedger.FEATURES; f++) {
            double value = slot.ledger()[f];
            sum[slot.target().seat()][f] += value;
            square[slot.target().seat()][f] += value * value;
          }
      }
      if (counted < replicates / 2) continue;
      tested++;
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
        for (int f = 0; f < EventLedger.FEATURES; f++) {
          double mean = sum[seat][f] / counted;
          double variance = square[seat][f] / counted - mean * mean;
          if (variance <= 1e-18) {
            Assert.assertEquals(mean, 0, 1e-9, "boundary " + b + " seat " + seat + " " + EventLedger.NAMES.get(f));
            continue;
          }
          double z = mean / Math.sqrt(variance / counted);
          Assert.assertTrue(
              Math.abs(z) < 4.5,
              "boundary " + b + " seat " + seat + " " + EventLedger.NAMES.get(f) + " mean=" + mean + " z=" + z);
        }
    }
    Assert.assertTrue(tested >= 16, "tested boundaries " + tested);
  }

  /** 立直の和了で終わった局の、和了を選んだ最後の境界の写し: engine の複製の元、その境界の選択、栄和か（立直の和了者の明細が Ron）。 */
  private record Win(GameEngine engine, EngineSelectionBuffer selections, boolean ron) {}

  /** 局の終わりの精算（局が終わっていなければ例外）。 */
  private static RoundSettlement settlement(GameStepResult step) {
    return switch (step) {
      case GameStepResult.RoundSettled settled -> settled.settlement();
      case GameStepResult.HanchanEnded ended -> ended.settlement();
      default -> throw new IllegalStateException("Unexpected boundary " + step);
    };
  }

  /**
   * 局末の裏ドラの項の条件付きの平均 0: 立直の和了（栄和 8 局・ツモ 8 局）で終わる局を集め、和了を選んだ最後の境界で複製して全員から見えない牌
   * だけを並べ替え（裏ドラの表示牌も引き直す）、同じ手を commit して局末の {@link EventLedger.Tracker#handEnd} の席ごとの ura を 2000 本で平均する。
   * 和了の手と重み ω は並べ替えで変わらないので、局ごと・席ごとに |z| &lt; 4.5。栄和の和了者の手牌は 13 枚（栄和の牌を足して数える）で、栄和と
   * ツモのどちらでも和了者の枠の ura は 0 でない（和了の手が数えられている）。
   */
  @Test(timeOut = 600_000)
  public void uraTermIsCentredUnderReshuffle() {
    var shapes = new HandShapeAnalyzer();
    List<Win> wins = new ArrayList<>();
    int rons = 0, tsumos = 0;
    for (long seed = 1; seed < 4000 && (rons < 8 || tsumos < 8); seed++) {
      var engine = new GameEngine(SeedMixer.mix64(seed * 32_452_843L));
      engine.getState().enableSeatStreamDraws();
      var random = new SplittableRandom(seed * 19);
      GameEngine last = null;
      EngineSelectionBuffer chosen = null;
      GameStepResult step = engine.stepHanchan();
      while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
        List<EngineDecisionPoint> points = awaiting.decisions();
        int[] picks = new int[points.size()];
        var selections = LedgerGames.select(engine.getState(), points, LedgerGames.PLAIN, random, shapes, picks);
        boolean win = false;
        for (int i = 0; i < points.size(); i++) {
          Action.Type type = points.get(i).legalActions().get(picks[i]).type();
          win |= type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI;
        }
        if (win) {
          last = engine.forkAtDecisionBoundary();
          chosen = selections;
        }
        step = engine.commitDecisions(selections);
      }
      if (last == null || !(settlement(step).result() instanceof RoundResult.Winning winning) || !winning.hasRiichiWinner())
        continue;
      boolean ron = false;
      for (WinClaim claim : winning.claims()) {
        if (!claim.hasRiichi()) continue;
        ron |= claim instanceof WinClaim.Ron;
        // 栄和の和了者の手牌は 13 枚のまま（handEnd は手番のイベントの栄和の牌を足す）
        if (claim instanceof WinClaim.Ron)
          Assert.assertEquals(engine.getState().hand(claim.winner()).effectiveTileCount(), 13, "ron winner of seed " + seed);
      }
      if (ron ? rons >= 8 : tsumos >= 8) continue;
      wins.add(new Win(last, chosen, ron));
      if (ron) rons++;
      else tsumos++;
    }
    Assert.assertEquals(rons, 8, "riichi ron wins");
    Assert.assertEquals(tsumos, 8, "riichi tsumo wins");
    int replicates = 2000, seats = GameState.NUM_PLAYERS;
    for (int b = 0; b < wins.size(); b++) {
      Win win = wins.get(b);
      double[] sum = new double[seats], square = new double[seats];
      int[] nonzero = new int[seats];
      int winner = -1;
      for (int r = 0; r < replicates; r++) {
        GameEngine fork = win.engine().forkAtDecisionBoundary();
        fork.getState().reshuffleUnseen(SeedMixer.mix64((b + 13L) * 1_000_000_009L + r));
        RoundSettlement settlement = settlement(fork.commitDecisions(win.selections()));
        var winning = (RoundResult.Winning) settlement.result();
        for (WinClaim claim : winning.claims()) if (claim.hasRiichi()) winner = claim.winner();
        List<Slot> slots = LedgerGames.slots();
        new EventLedger.Tracker().handEnd(fork.getState(), settlement, slots);
        for (Slot slot : slots) {
          int seat = slot.target().seat();
          double value = slot.ledger()[EventLedger.URA];
          for (int f = 0; f < EventLedger.FEATURES; f++)
            if (f != EventLedger.URA)
              Assert.assertEquals(slot.ledger()[f], 0.0, "win " + b + " seat " + seat + " " + EventLedger.NAMES.get(f));
          sum[seat] += value;
          square[seat] += value * value;
          if (value != 0) nonzero[seat]++;
        }
      }
      Assert.assertTrue(
          nonzero[winner] > replicates / 2,
          "win " + b + " (ron " + win.ron() + ") winner " + winner + " ura nonzero in " + nonzero[winner] + " replicates");
      for (int seat = 0; seat < seats; seat++) {
        double mean = sum[seat] / replicates, variance = square[seat] / replicates - mean * mean;
        if (variance <= 1e-18) {
          Assert.assertEquals(mean, 0, 1e-9, "win " + b + " seat " + seat);
          continue;
        }
        double z = mean / Math.sqrt(variance / replicates);
        Assert.assertTrue(
            Math.abs(z) < 4.5, "win " + b + " (ron " + win.ron() + ") seat " + seat + " mean=" + mean + " z=" + z);
      }
    }
  }

  /**
   * 境界の種類（TURN でツモの直後）: 0 = 聴牌の他家がいる、1 = 立直の席がいる、2 = 遅れた表示を待つ（嶺上のツモで打牌の後に表示）、3 = 暗槓を
   * 選べる。どれでもなければ -1。
   */
  private static int category(GameState state, EngineDecisionPoint point, HandShapeAnalyzer shapes) {
    var draw = (TurnEvent.Draw) state.turnEvent();
    if (draw.doraRevealPending()) return 2;
    for (Action action : point.legalActions()) if (action.type() == Action.Type.ANKAN) return 3;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
      if (seat != point.player() && state.isRiichi(seat)) return 1;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
      if (seat != point.player() && shapes.calculateMinimum(state.hand(seat)) == 0) return 0;
    return -1;
  }
}
