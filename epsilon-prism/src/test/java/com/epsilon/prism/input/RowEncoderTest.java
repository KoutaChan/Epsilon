package com.epsilon.prism.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import java.nio.ByteBuffer;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 行ライターを長さの異なるバッチの間で使い回し、各判断を単独で書いたバイト列と照合する。 */
public class RowEncoderTest {
  @Test(dataProvider = "formats", dataProviderClass = InputFormatTest.class)
  public void switchingBuffersPreservesEveryRow(InputFormat format) {
    var capture = new RowEncoder(format);
    var writer = format.writer();
    var first = HostInputs.allocate(format, RowEncoder.ONE_ROW, ByteBuffer::allocate);
    var second = HostInputs.allocate(format, RowEncoder.ONE_ROW, ByteBuffer::allocateDirect);
    int[] rows = {0};
    new TableWalk(format.variant(), TableWalk.Policy.CALLS)
        .run(
            739,
            1,
            (engine, slot) -> {
              Observation observation = capture.encode(engine, slot);
              var host = (rows[0]++ & 1) == 0 ? first : second;
              var plan = host.plan();
              plan.clear();
              Assert.assertEquals(plan.add(observation), 0);
              plan.seal();
              writer.write(observation, host, 0);
              int bytes = capture.batch.bytes();
              Assert.assertEquals(host.input().bytes(), bytes);
              for (var tensor : InputBlob.Tensor.values()) {
                int start = plan.offset(tensor);
                int length =
                    plan.count(tensor.axis()) * format.width(tensor) * tensor.element().bytes();
                var actual = host.input().blob().slice(start, length);
                var expected = capture.batch.blob().slice(start, length);
                Assert.assertEquals(actual.mismatch(expected), -1, tensor.name());
              }
              for (int action = 0; action < observation.legalActions(); action++) {
                Assert.assertEquals(
                    host.input().priorShanten(action), capture.batch.priorShanten(action));
                Assert.assertEquals(
                    host.input().priorUkeire(action), capture.batch.priorUkeire(action));
              }
            });
    Assert.assertTrue(rows[0] > 100);
  }

  @Test(dataProvider = "variants", dataProviderClass = InputFormatTest.class)
  public void aDifferentBatchFormatIsRefusedBeforeWriting(Variant variant) {
    var base = InputFormat.of(variant);
    var writer = InputFormat.of(variant, Set.of(InputFeature.CANDIDATE_SEGMENTS)).writer();
    var capture = new RowEncoder(base);
    var host = HostInputs.allocate(base, RowEncoder.ONE_ROW, ByteBuffer::allocate);
    var scene = PlayedScene.first(variant).start();
    var observation = capture.capture(scene.engine, 0);
    host.plan().add(observation);
    host.plan().seal();
    Assert.expectThrows(IllegalArgumentException.class, () -> writer.write(observation, host, 0));
    Assert.assertEquals(
        host.input().blob().slice(0, host.input().bytes()),
        ByteBuffer.allocate(host.input().bytes()));
  }
}
