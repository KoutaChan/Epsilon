package com.epsilon.nano.ai.decision.benchmark;

import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayCorpus.BucketStatistics;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayCorpus.Corpus;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayCorpus.Role;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 推論再実行用データをビッグエンディアンで保存し、NaN のビット列も保持することを検証する。 */
public class EpsilonDecisionInferenceReplayCorpusStoreTest {
  @Test
  public void inferenceRowsKeepCanonicalBigEndianPayloadAndRawNanBits() throws Exception {
    GameEngine engine = new GameEngine(8181L);
    var boundary = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var point = boundary.decisions().getFirst();
    var bucket = DecisionBatchBuilder.selectInferenceBucket(point.legalActions());
    var builder = DecisionBatchBuilder.inference(1, bucket);
    builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
    DecisionHostBatch batch = builder.build();
    batch
        .inputs()
        .writer(0)
        .round(DecisionInputSchema.RoundFloat.HONBA, Float.intBitsToFloat(0x7fc01234));
    batch.inputs().writer(0).round(DecisionInputSchema.RoundFloat.KYOTAKU, -0.0f);
    var rows = batch.sliceRows(0, 1);
    ShortBuffer categories = ShortBuffer.allocate(rows.inputCategoricalElementCount());
    FloatBuffer numerics = FloatBuffer.allocate(rows.inputNumericElementCount());
    rows.copyInputCategoriesTo(categories);
    rows.copyInputNumericsTo(numerics);
    Corpus actor = corpus(Role.ACTOR_POLICY_AND_VALUE, batch);
    Corpus opponent = corpus(Role.OPPONENT_POLICY_ONLY, batch);
    Path directory = Files.createTempDirectory("pico-replay-corpus-");
    Path file = directory.resolve("corpus.bin");
    try {
      var restored =
          EpsilonDecisionInferenceReplayCorpusStore.create(file, actor, opponent, 23L, 2);
      Assert.assertEquals(
          Files.readAllBytes(file), expectedBytes(batch, categories.array(), numerics.array()));
      for (Corpus corpus : new Corpus[] {restored.actor(), restored.opponent()}) {
        var input = corpus.storedBatch(bucket).inputs();
        Assert.assertEquals(input.denseCategories(), categories.array());
        float[] actual = input.denseNumerics();
        Assert.assertEquals(actual.length, numerics.capacity());
        for (int index = 0; index < actual.length; index++) {
          Assert.assertEquals(
              Float.floatToRawIntBits(actual[index]),
              Float.floatToRawIntBits(numerics.get(index)),
              "numeric index=" + index);
        }
        Assert.assertEquals(
            Float.floatToRawIntBits(
                corpus.storedBatch(bucket).roundNumeric(0, DecisionInputSchema.RoundFloat.HONBA)),
            0x7fc01234);
      }
    } finally {
      Files.deleteIfExists(file);
      Files.deleteIfExists(directory);
    }
  }

  private static Corpus corpus(Role role, DecisionHostBatch batch) {
    return Corpus.restore(
        role,
        Map.of(batch.bucket(), batch),
        "FIXED_TEST",
        Map.of(batch.bucket(), new BucketStatistics(3L, 1, 2)));
  }

  private static byte[] expectedBytes(DecisionHostBatch batch, short[] categories, float[] numerics)
      throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (DataOutputStream out = new DataOutputStream(bytes)) {
      out.writeLong(0x4550535245504C59L);
      out.writeInt(2);
      writeText(out, DecisionInputSchema.fingerprint());
      out.writeLong(23L);
      out.writeInt(2);
      out.writeInt(2);
      for (Role role : new Role[] {Role.ACTOR_POLICY_AND_VALUE, Role.OPPONENT_POLICY_ONLY}) {
        writeText(out, role.name());
        writeText(out, "FIXED_TEST");
        out.writeInt(1);
        out.writeInt(batch.bucket().legalActionCapacity());
        out.writeInt(batch.bucket().actionTransitionCapacity());
        out.writeLong(3L);
        out.writeInt(1);
        out.writeInt(2);
        out.writeInt(categories.length);
        out.writeInt(numerics.length);
        for (short value : categories) out.writeShort(value);
        for (float value : numerics) out.writeInt(Float.floatToRawIntBits(value));
      }
    }
    return bytes.toByteArray();
  }

  private static void writeText(DataOutputStream out, String value) throws IOException {
    byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
    out.writeInt(bytes.length);
    out.write(bytes);
  }
}
