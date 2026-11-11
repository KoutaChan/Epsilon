package com.epsilon.prism4p;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism4p.input.InputSchema;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 四人打ちの形式のフィンガープリントを、公開した重みの model.json と保存したコホートのマニフェストに入っている値に固定する。公開した重みと
 * 保存したコホートが読めるかは、この値で決まる。補助のヘッドのフィンガープリントは四人打ちに固有の値を含むので、このモジュールで固定する。
 */
public class FourPlayerFingerprintTest {

  @Test
  public void formatFingerprintsStayTheSame() {
    Assert.assertEquals(InputSchema.fingerprint(), "eeeb4f583cd2b92a");
    Assert.assertEquals(TargetSchema.of(Variant.TENHOU_4P).fingerprint(), "391be70aa4a7bf7b");
  }
}
