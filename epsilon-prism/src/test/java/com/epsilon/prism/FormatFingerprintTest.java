package com.epsilon.prism;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 観測と補助の教師の形式を既存のフィンガープリントと照合する。 */
public class FormatFingerprintTest {

  @Test
  public void formatFingerprintsStayTheSame() {
    Assert.assertEquals(OutcomeCodec.of(Variant.TENHOU_4P).fingerprint(), "a71ba8e0c3e8c105");
    Assert.assertEquals(OracleCodec.of(Variant.TENHOU_4P).fingerprint(), "77275bf2f7b5ae35");
  }

  /** 三麻の補助の教師の幅とフィンガープリントを確かめる。 */
  @Test
  public void threePlayerAuxiliaryFormatsDiffer() {
    Assert.assertNotEquals(
        OutcomeCodec.of(Variant.TENHOU_3P).fingerprint(),
        OutcomeCodec.of(Variant.TENHOU_4P).fingerprint());
    Assert.assertNotEquals(
        OracleCodec.of(Variant.TENHOU_3P).fingerprint(),
        OracleCodec.of(Variant.TENHOU_4P).fingerprint());
    Assert.assertEquals(OracleCodec.of(Variant.TENHOU_3P).opponents(), 2);
    TargetSchema three = TargetSchema.of(Variant.TENHOU_3P);
    Assert.assertEquals(three.width(TargetSchema.CandidateTarget.WINNER), 2);
    Assert.assertEquals(
        three.candidateWidth(), TargetSchema.of(Variant.TENHOU_4P).candidateWidth() - 1);
    Assert.assertEquals(three.fingerprint(), "396d676ebb8abdb3");
  }

  /** 四人打ちの観測（O）の形式は、以前の PRISM と同じフィンガープリントを持つ。 */
  @Test
  public void observationKeepsTheOldFingerprint() {
    Assert.assertEquals(
        com.epsilon.agent.observe.ObservationSchema.of(Variant.TENHOU_4P).fingerprint(),
        "bb3db98cb8b5b43a");
  }
}
