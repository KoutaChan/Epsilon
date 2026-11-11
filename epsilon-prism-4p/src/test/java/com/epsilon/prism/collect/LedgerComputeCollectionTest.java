package com.epsilon.prism.collect;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerRecording;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 帳簿で計算する組を絞って集めた行を、全ての組を計算して集めた行と比べる（分岐・写し・根の記録を含む、本番と同じ収集の処理を通す）。帳簿の
 * ほかはビットまで一致する。計算した組の列は同じ値（同じ乱数で fp16 に丸めた値）になり、計算しない組の列は、測った手では 0、測っていない手
 * では NaN になる。
 */
public class LedgerComputeCollectionTest {
  private static final LedgerColumn[] COLUMNS = LedgerColumn.values();

  /** ラベル行、ロールイン、写しの中の行がどれも出るようにした抽出率（{@link CollectorTest} と同じ）。 */
  private static final Sampling SAMPLING =
      TestSampling.grouped(0.25, 0.03, 0.03, 0.03, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  /** 一様分布を ε_b = 0.05 だけ混ぜる行動方策。 */
  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE);

  @Test(timeOut = 900_000)
  public void narrowerComputeKeepsTheComputedColumnsAndZeroesTheRest() {
    List<Row> full = collect(CollectorTest.LEDGER);
    Assert.assertFalse(full.isEmpty());
    List<Set<LedgerGroup>> subsets =
        List.of(
            Set.of(LedgerGroup.IMPROVE),
            Set.of(LedgerGroup.POINTS),
            Set.of(LedgerGroup.DORA),
            Set.of(LedgerGroup.URA),
            Set.of(LedgerGroup.POLICY),
            EnumSet.of(LedgerGroup.POINTS, LedgerGroup.URA, LedgerGroup.POLICY));
    for (Set<LedgerGroup> compute : subsets) {
      List<Row> part = collect(new LedgerRecording(compute, 32, 8));
      Assert.assertEquals(projected(part, compute, false), projected(full, compute, true), compute.toString());
    }
  }

  /** 半荘 200〜203 を集め、確定して読み戻した行。 */
  private static List<Row> collect(LedgerRecording ledger) {
    return CollectorTest.collectGames(
            SAMPLING, ledger, 7_001L * 5, 200, 4, FLOOR, new CollectorTest.HashedScorePolicy(), 1, 4)
        .rows();
  }

  /**
   * 行ごとに、帳簿を外して直列化した行と、帳簿を計算する組だけに絞って書いた文字列をつなぎ、並べ替えた一覧を返す。計算する組の列には値の
   * ビットを書く。測った手のほかの列には 0 の印を書く（全ての組を計算した行では値を見ない。絞った行では、0 でなければ値のビットを書く）。
   * 測っていない手には NaN の印を書く。
   *
   * @param rows 行
   * @param compute 帳簿で計算する組
   * @param reference 全ての組を計算した行なら true
   */
  private static List<String> projected(List<Row> rows, Set<LedgerGroup> compute, boolean reference) {
    RowCodec codec = new RowCodec(0, TENHOU_4P);
    List<String> result = new ArrayList<>();
    for (Row row : rows) {
      Assert.assertEquals(row.ledgerFeatures(), LedgerColumn.COUNT);
      StringBuilder text = new StringBuilder(Base64.getEncoder().encodeToString(codec.encode(stripped(row))));
      for (int a = 0; a < row.actions(); a++)
        for (int f = 0; f < COLUMNS.length; f++) {
          float value = row.ledger(a, f);
          text.append(',');
          if (!row.measured(a)) {
            Assert.assertTrue(Float.isNaN(value), "unmeasured action " + a);
            text.append('n');
          } else if (compute.contains(COLUMNS[f].group())) {
            text.append(Integer.toHexString(Float.floatToRawIntBits(value)));
          } else if (reference || value == 0f) {
            text.append('z');
          } else {
            text.append("x").append(Integer.toHexString(Float.floatToRawIntBits(value)));
          }
        }
      result.add(text.toString());
    }
    Collections.sort(result);
    return result;
  }

  /** 帳簿を外した行。 */
  private static Row stripped(Row row) {
    return new Row(
        row.gameId(),
        row.decisionId(),
        row.decisionClass(),
        row.purpose(),
        row.source(),
        row.played(),
        row.values(),
        row.prior(),
        row.support(),
        row.deviation(),
        row.deviationProbability(),
        row.propensity(),
        row.rootDecisionId(),
        new float[0],
        row.outcomes(),
        row.oracle(),
        row.observation());
  }
}
