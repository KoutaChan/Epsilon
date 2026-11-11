package com.epsilon.prism3p;

import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism3p.input.InputSchema;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの形式のフィンガープリントを定数で固定する。保存した三人打ちのネットワークとコホートを読めるかはこの値で決まるので、フィールドや
 * 語彙を変えたらこのテストで気付く。補助のヘッドのフィンガープリントは三人打ちの入力の定義を読んで決まるので、このモジュールで固定する。
 */
public class ThreePlayerFingerprintTest {

  @Test
  public void formatFingerprintsStayTheSame() {
    Assert.assertEquals(InputSchema.fingerprint(), "f22b982987b2d084");
    Assert.assertEquals(InputSchema.FORMAT.fingerprint(), "f22b982987b2d084");
    Assert.assertEquals(TargetSchema.of(Variant.TENHOU_3P).fingerprint(), "396d676ebb8abdb3");
    Assert.assertEquals(ObservationSchema.of(Variant.TENHOU_3P).fingerprint(), "66529273ed75ef58");
    Assert.assertEquals(OutcomeCodec.of(Variant.TENHOU_3P).fingerprint(), "758353849c4267c2");
    Assert.assertEquals(OracleCodec.of(Variant.TENHOU_3P).fingerprint(), "f1fa243f4cbc571a");
  }

  @Test
  public void theInputDiffersFromTheFourPlayerOne() {
    Assert.assertNotEquals(InputSchema.fingerprint(), "eeeb4f583cd2b92a");
  }

  /** 順位の効用は天鳳の表（三人打ちで +1・0・−1）を使う。効用は入力の値（待ちの射影と、ゲートに直接渡す値）に入るので、フィンガープリントにも入る。 */
  @Test
  public void theUtilityValuesAreTheOwnerTable() {
    Assert.assertSame(InputSchema.UTILITY, RankUtility.TENHOU);
    Assert.assertEquals(RankUtility.TENHOU.value(Variant.TENHOU_3P, 0), 1.0);
    Assert.assertEquals(RankUtility.TENHOU.value(Variant.TENHOU_3P, 1), 0.0);
    Assert.assertEquals(RankUtility.TENHOU.value(Variant.TENHOU_3P, 2), -1.0);
  }
}
