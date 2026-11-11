package com.epsilon.prism.luck;

import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.noise.LedgerGroup;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 帳簿の計算器が、計算する組と切る幅を収集の値（{@link LedgerRecording}）から読むことを、同じ局に計算器を並べて確かめる。計算する組の列は、
 * 全ての組を計算したときとビット単位で一致し、ほかの列は 0 のまま。点数の列を切る幅を変えると点数の列だけが変わり、方策の列で s を切る幅を
 * 変えると方策の列だけが変わる。
 */
public class LedgerComputeTest {
  private static final LedgerColumn[] COLUMNS = LedgerColumn.values();

  /** 全ての組を計算する帳簿と、比べる帳簿（組の部分集合 7 つと、切る幅を変えた 2 つ）。 */
  private static final List<LedgerRecording> RECORDINGS =
      List.of(
          LedgerGames.RECORDING,
          new LedgerRecording(Set.of(LedgerGroup.IMPROVE), 32, 8),
          new LedgerRecording(Set.of(LedgerGroup.POINTS), 32, 8),
          new LedgerRecording(Set.of(LedgerGroup.DORA), 32, 8),
          new LedgerRecording(Set.of(LedgerGroup.URA), 32, 8),
          new LedgerRecording(Set.of(LedgerGroup.POLICY), 32, 8),
          new LedgerRecording(EnumSet.of(LedgerGroup.IMPROVE, LedgerGroup.DORA), 32, 8),
          new LedgerRecording(EnumSet.of(LedgerGroup.POINTS, LedgerGroup.URA, LedgerGroup.POLICY), 32, 8),
          new LedgerRecording(LedgerGroup.computed(), 2, 8),
          new LedgerRecording(LedgerGroup.computed(), 32, 0.25));

  /** 点数の列を切る幅を変えた帳簿と、方策の列で s を切る幅を変えた帳簿の番号。 */
  private static final int POINT_CLIP = 8, POLICY_CLIP = 9;

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /**
   * 席別ツモ列の半荘を続けて打ち（ふつうの打ち方で 300 局と、槓を選ぶ打ち方で 200 局）、判断の時点ごとに全ての計算器で {@code observe} と
   * {@code policy} を呼び、局末に {@code handEnd} を呼ぶ。局末には、席ごとの帳簿を全ての組を計算した帳簿と比べる。主な列では、全ての組を
   * 計算した帳簿で 0 でない値が 10 回以上出ることも確かめる。
   */
  @Test(dataProvider = "variants", timeOut = 600_000)
  public void eachGroupIsComputedAsInTheFullLedger(Variant variant) {
    long[][] differences = new long[RECORDINGS.size()][COLUMNS.length];
    long[] nonzero = new long[COLUMNS.length];
    LedgerGames.walk(variant, 51L, LedgerGames.PLAIN, 300, new Side(variant, differences, nonzero));
    LedgerGames.walk(variant, 52L, LedgerGames.KANS, 200, new Side(variant, differences, nonzero));
    for (int k = 1; k < RECORDINGS.size(); k++) {
      LedgerRecording recording = RECORDINGS.get(k);
      LedgerGroup changing = k == POINT_CLIP ? LedgerGroup.POINTS : k == POLICY_CLIP ? LedgerGroup.POLICY : null;
      long changed = 0;
      for (LedgerColumn column : COLUMNS) {
        long count = differences[k][column.ordinal()];
        if (column.group() == changing) changed += count;
        else Assert.assertEquals(count, 0L, variant + " " + recording + " " + column.reportName());
      }
      if (changing != null)
        Assert.assertTrue(changed > 0, variant + " " + recording + " changed no " + changing + " column");
    }
    for (LedgerColumn column :
        new LedgerColumn[] {
          LedgerColumn.OWN_IMPROVE_1,
          LedgerColumn.OWN_TSUMO,
          LedgerColumn.OPP_TSUMO,
          LedgerColumn.DORA_SELF,
          LedgerColumn.URA,
          LedgerColumn.POLICY_SELF,
          LedgerColumn.POLICY_OPPONENT
        })
      Assert.assertTrue(
          nonzero[column.ordinal()] >= 10, variant + " " + column.reportName() + " " + nonzero[column.ordinal()]);
  }

  /** 全ての帳簿の計算器を同じ局に並べる歩き手。局ごとに計算器と受け取り先を作り直す。 */
  private static final class Side implements LedgerGames.Visitor {
    private final Variant variant;
    private final long[][] differences;
    private final long[] nonzero;
    private final LedgerTracker[] trackers = new LedgerTracker[RECORDINGS.size()];
    private final List<List<LedgerGames.Seat>> seats = new ArrayList<>();

    Side(Variant variant, long[][] differences, long[] nonzero) {
      this.variant = variant;
      this.differences = differences;
      this.nonzero = nonzero;
      start();
    }

    private void start() {
      seats.clear();
      for (int k = 0; k < trackers.length; k++) trackers[k] = new LedgerTracker(RECORDINGS.get(k));
    }

    private List<LedgerGames.Seat> targets(int k) {
      return seats.isEmpty() ? List.of() : seats.get(k);
    }

    @Override
    public void decision(Engine engine, RuleWorkspace ws) {
      FullView full = engine.full();
      for (int k = 0; k < trackers.length; k++) trackers[k].observe(full, targets(k), ws);
      if (seats.isEmpty()) for (int k = 0; k < trackers.length; k++) seats.add(LedgerGames.seats(variant));
      Decisions decisions = engine.decisions();
      for (int slot = 0; slot < decisions.size(); slot++) {
        int n = decisions.legalCount(slot), seat = decisions.seat(slot);
        if (n < 2) continue;
        float[] scores = new float[n], p = new float[n];
        long code = 0;
        for (int a = 0; a < n; a++) {
          code = SeedMixer.mix64(code ^ decisions.legal(slot, a));
          scores[a] = (float) (2 * CrnSeeds.unit(SeedMixer.mix64(decisions.legal(slot, a) * 131L + seat)) - 1);
          p[a] = 1f / n;
        }
        int chosen = (int) Long.remainderUnsigned(code, n);
        for (int k = 0; k < trackers.length; k++) trackers[k].policy(seat, scores, p, chosen, n, targets(k));
      }
    }

    @Override
    public void roundEnd(Engine engine) {
      FullView full = engine.full();
      RoundEnd end = engine.roundEnd();
      for (int k = 0; k < trackers.length; k++) trackers[k].handEnd(full, end, targets(k));
      if (!seats.isEmpty()) compare();
      start();
    }

    /** 席ごとの帳簿を、全ての組を計算した帳簿と比べる。計算しない組の列は 0 のまま。 */
    private void compare() {
      List<LedgerGames.Seat> reference = seats.get(0);
      for (int seat = 0; seat < reference.size(); seat++) {
        double[] expected = reference.get(seat).ledger();
        for (int f = 0; f < COLUMNS.length; f++) if (expected[f] != 0) nonzero[f]++;
        for (int k = 1; k < trackers.length; k++) {
          Set<LedgerGroup> compute = RECORDINGS.get(k).compute();
          double[] actual = seats.get(k).get(seat).ledger();
          for (int f = 0; f < COLUMNS.length; f++) {
            if (!compute.contains(COLUMNS[f].group())) {
              Assert.assertEquals(
                  Double.doubleToRawLongBits(actual[f]),
                  0L,
                  variant + " " + RECORDINGS.get(k) + " computed " + COLUMNS[f].reportName() + " = " + actual[f]);
              continue;
            }
            if (Double.doubleToRawLongBits(actual[f]) != Double.doubleToRawLongBits(expected[f])) differences[k][f]++;
          }
        }
      }
    }
  }
}
