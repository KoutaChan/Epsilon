package com.epsilon.prism;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * ルールから配置を作る形式のフィンガープリントが、既存の model.json とコホートのマニフェストに書かれた値から変わらないことを確かめる。
 * snapshot やコホートを読めるかは、この値で決まる。四人打ちの入力と補助のヘッドの形式は、epsilon-prism-4p のテストで
 * 固定する。
 */
public class FormatFingerprintTest {

  @Test
  public void formatFingerprintsStayTheSame() {
    Assert.assertEquals(OutcomeCodec.of(Variant.TENHOU_4P).fingerprint(), "a71ba8e0c3e8c105");
    Assert.assertEquals(OracleCodec.of(Variant.TENHOU_4P).fingerprint(), "77275bf2f7b5ae35");
  }

  /**
   * 三人打ちの局末の記録と Oracle の記録は、席の数から配置を作るので、四人打ちとはフィンガープリントが違う。三人打ちには入力の形式が
   * 無いので、補助のヘッドのフィンガープリントは作れない。
   */
  @Test
  public void threePlayerAuxiliaryFormatsDiffer() {
    Assert.assertNotEquals(
        OutcomeCodec.of(Variant.TENHOU_3P).fingerprint(), OutcomeCodec.of(Variant.TENHOU_4P).fingerprint());
    Assert.assertNotEquals(
        OracleCodec.of(Variant.TENHOU_3P).fingerprint(), OracleCodec.of(Variant.TENHOU_4P).fingerprint());
    Assert.assertEquals(OracleCodec.of(Variant.TENHOU_3P).opponents(), 2);
    TargetSchema three = TargetSchema.of(Variant.TENHOU_3P);
    Assert.assertEquals(three.width(TargetSchema.CandidateTarget.WINNER), 2);
    Assert.assertEquals(
        three.candidateWidth(), TargetSchema.of(Variant.TENHOU_4P).candidateWidth() - 1);
    Assert.expectThrows(RuntimeException.class, three::fingerprint);
  }

  /** 四人打ちの観測（O）の形式は、以前の PRISM と同じフィンガープリントを持つ。 */
  @Test
  public void observationKeepsTheOldFingerprint() {
    Assert.assertEquals(
        com.epsilon.agent.observe.ObservationSchema.of(Variant.TENHOU_4P).fingerprint(), "bb3db98cb8b5b43a");
  }
}
