package com.epsilon.prism.run;

import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.luck.LedgerCalibrator;
import com.epsilon.prism.luck.LedgerCatalog;
import com.epsilon.prism.luck.LuckSettlement;
import com.epsilon.prism.noise.LedgerCalibrationKind;
import com.epsilon.prism.noise.LedgerCalibrationSource;
import com.epsilon.prism.noise.LedgerCells;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.settings.PrismSettings;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 持ち越しのキーのうち、帳簿の部分（{@link com.epsilon.prism.luck.LedgerMeaning}）のテスト。帳簿の列や教師の意味を決める値が
 * 変わると、持ち越しを捨て、変わった項目を理由に書く。そうした値は、列の並び、モード、計算する組、クリップの幅、傾きの列の値、補正に
 * 使う組、教師に掛ける補正率の形。当てはめにだけ効く値（ridge、列 1 つに要る行の数、偏りの警告のしきい値、裏ドラの監査の割合、補正率の
 * 格子と罰の強さ）が変わっても捨てない。
 * state.json に書いて読み戻したキーは、同じ設定から作ったキーと等しい。
 */
public class CarryKeyTest {

  /** TEMPO のゲインを補正の R² で上げない設定。補正しないモードではこれが要る。 */
  private static final String NO_GAIN = "noise.tempo.noiseGain=NONE";

  /** 設定から作るキー。{@link CarryKey#of} と同じ値で、Φ は点数だけの Φ とする。 */
  private static CarryKey key(String... assignments) {
    PrismSettings settings = TestSettings.of(assignments);
    return new CarryKey(
        LuckSettlement.of(settings.noise(), settings.run().variant()).meaning(), true);
  }

  @Test
  public void theSameLedgerKeepsTheCarryThroughTheState() {
    List<String[]> runs =
        List.of(
            new String[0],
            new String[] {"noise.ledger.mode=OFF", NO_GAIN},
            new String[] {"noise.ledger.mode=AUDIT", NO_GAIN});
    for (String[] settings : runs) {
      String label = String.join(" ", settings);
      CarryKey key = key(settings);
      Assert.assertEquals(key(settings), key, label);
      Assert.assertNull(key(settings).resetReason(key, true), label);
      PrismCarry carry = new PrismCarry(0.25, key);
      PrismCarry back =
          RecordFile.read(PrismCarry.class, RecordFile.json(carry), "state.json").value();
      Assert.assertEquals(back, carry, label);
      Assert.assertNull(key(settings).resetReason(back.key(), true), label);
    }
  }

  @Test
  public void aChangedLedgerMeaningDropsTheCarry() {
    CarryKey base = key();
    Object[][] changes = {
      {
        "compute",
        new String[] {
          "noise.ledger.compute=[IMPROVE, POINTS, DORA, POLICY]",
          "noise.ledger.apply=[IMPROVE, POINTS, DORA, POLICY]"
        }
      },
      {"pointClip", new String[] {"noise.ledger.pointClip=16"}},
      {"policyClip", new String[] {"noise.ledger.policyClip=4"}},
      {"slopeStep", new String[] {"noise.ledger.slopeStep=2000"}},
      {"slopeRefresh", new String[] {"noise.ledger.slopeRefresh=HAND"}},
      {"apply", new String[] {"noise.ledger.apply=[POINTS, POLICY]"}},
      {"mode", new String[] {"noise.ledger.mode=AUDIT", NO_GAIN}},
      {"columns", new String[] {"noise.ledger.mode=OFF", NO_GAIN}}
    };
    for (Object[] change : changes) {
      CarryKey changed = key((String[]) change[1]);
      String reason = changed.resetReason(base, true);
      Assert.assertNotNull(reason, (String) change[0]);
      Assert.assertTrue(reason.startsWith("the ledger " + change[0] + " changed"), reason);
      Assert.assertNotNull(base.resetReason(changed, true), (String) change[0]);
    }
  }

  /**
   * 帳簿の補正率を教師に使わない SHADOW は、補正率の節が無い設定と同じキーになる。APPLY は補正率の形がキーに入るので、OFF・SHADOW との
   * 間で持ち越しを捨て、理由に補正率を書く。APPLY のまま格子や罰の強さを変えても捨てない。次の反復の補正率を持つ持ち越しは、state.json に
   * 書いて読み戻すと同じ値になる。
   */
  @Test
  public void onlyAnAppliedCalibrationChangesTheKey() {
    CarryKey base = key();
    CarryKey shadow = key("noise.ledger.calibration.mode=SHADOW");
    CarryKey apply = key("noise.ledger.calibration.mode=APPLY");
    Assert.assertEquals(shadow, base);
    Assert.assertNull(shadow.resetReason(base, true));
    Assert.assertEquals(apply.ledger().calibration(), LedgerCalibrationKind.COMMON_SCALE);
    for (CarryKey other : List.of(base, shadow)) {
      Assert.assertEquals(
          apply.resetReason(other, true),
          "the ledger calibration changed from null to COMMON_SCALE");
      Assert.assertEquals(
          other.resetReason(apply, true),
          "the ledger calibration changed from COMMON_SCALE to null");
    }
    for (String change :
        List.of(
            "noise.ledger.calibration.gapGrid=5",
            "noise.ledger.calibration.roundGrid=2",
            "noise.ledger.calibration.shrink=0",
            "noise.ledger.calibration.regularization=0.01")) {
      Assert.assertNull(
          key("noise.ledger.calibration.mode=APPLY", change).resetReason(apply, true), change);
    }
    List<Double> coefficients = new ArrayList<>();
    for (int j = 0; j < 108; j++) coefficients.add(j / 107.0);
    var calibrator =
        new LedgerCalibrator(
            LedgerCalibrationKind.COMMON_SCALE,
            LedgerCalibrationSource.PREVIOUS_COHORT,
            Variant.TENHOU_4P,
            3,
            LedgerCatalog.of(Variant.TENHOU_4P).columns(),
            LedgerGroup.computed(),
            LedgerCells.CLASS_STAGE_RANK,
            9,
            3,
            1e-4,
            0.08,
            1234,
            coefficients);
    PrismCarry carry = new PrismCarry(0.25, apply, calibrator);
    PrismCarry back =
        RecordFile.read(PrismCarry.class, RecordFile.json(carry), "state.json").value();
    Assert.assertEquals(back, carry);
    Assert.assertNull(key("noise.ledger.calibration.mode=APPLY").resetReason(back.key(), true));
    PrismCarry plain = new PrismCarry(0.25, base);
    Assert.assertFalse(RecordFile.json(plain).contains("calibrat"), RecordFile.json(plain));
  }

  @Test
  public void fitOnlyValuesKeepTheCarry() {
    CarryKey base = key();
    for (String change :
        List.of(
            "noise.ledger.ridge=0.01",
            "noise.ledger.minRowsPerColumn=8",
            "noise.ledger.biasZ=4",
            "noise.ura.sample=0.2")) {
      Assert.assertNull(key(change).resetReason(base, true), change);
    }
  }
}
