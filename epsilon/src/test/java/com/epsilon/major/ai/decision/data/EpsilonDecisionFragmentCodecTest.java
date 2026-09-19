package com.epsilon.major.ai.decision.data;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.ByteBuffer;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Decision学習fragmentが現行形式以外を明示的に拒否することを検証する。 */
public class EpsilonDecisionFragmentCodecTest {

  @Test
  public void v46FragmentMagicIsRejectedBeforeReadingPayload() {
    byte[] oldHeader = ByteBuffer.allocate(Integer.BYTES).putInt(0xED10_0023).array();
    Assert.expectThrows(
        EpsilonDecisionFragmentCodec.UnsupportedFormatException.class,
        () -> EpsilonDecisionFragmentCodec.readHeaderMetadata(new ByteArrayInputStream(oldHeader)));
  }

  @Test
  public void mismatchedSchemaIsRejectedBeforeReadingPayload() throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (DataOutputStream out = new DataOutputStream(bytes)) {
      out.writeInt(0xED10_0024);
      out.writeUTF("not-the-current-schema");
    }
    Assert.expectThrows(
        EpsilonDecisionFragmentCodec.UnsupportedFormatException.class,
        () ->
            EpsilonDecisionFragmentCodec.readHeaderMetadata(
                new ByteArrayInputStream(bytes.toByteArray())));
  }
}
