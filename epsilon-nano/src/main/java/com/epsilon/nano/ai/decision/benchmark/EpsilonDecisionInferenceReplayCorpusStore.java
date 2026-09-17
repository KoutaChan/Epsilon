package com.epsilon.nano.ai.decision.benchmark;

import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayCorpus.BucketStatistics;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayCorpus.Corpus;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayCorpus.Report;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionInferenceReplayCorpus.Role;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Map;

/** 学習側と対戦相手側の推論から採取した固定入力データを保存・復元する。 */
public final class EpsilonDecisionInferenceReplayCorpusStore {

  private static final long MAGIC = 0x4550535245504C59L;
  private static final int FORMAT_VERSION = 2;
  private static final int ROLE_COUNT = 2;
  private static final int MAXIMUM_BUCKETS = 32;
  private static final int MAXIMUM_TEXT_BYTES = 256;

  private EpsilonDecisionInferenceReplayCorpusStore() {}

  /** 永続化した二役割の固定入力データ。 */
  public record Bundle(
      Corpus actor,
      Corpus opponent,
      int formatVersion,
      String schemaFingerprint,
      long seedBase,
      int captureGames) {}

  /**
   * 二役割の固定入力データを新規ファイルへ逐次処理書込する。
   *
   * <p>同じディレクトリの一時ファイルを完成させてからハードリンクで公開する。既存ファイルは置換せず、同じA/B名で固定入力データを誤って作り直す操作を拒否する。
   *
   * @param file 新規作成する固定入力データファイル
   * @param actor 方策モデル方策・価値固定入力データ
   * @param opponent 対戦相手方策固定入力データ
   * @return 公開するファイルを復号・検証して復元した保存データ一式
   * @throws IOException ファイル作成または検証に失敗した場合
   */
  static Bundle create(Path file, Corpus actor, Corpus opponent, long seedBase, int captureGames)
      throws IOException {
    Path target = file.toAbsolutePath().normalize();
    Path parent = target.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    Path temporary =
        Files.createTempFile(
            parent == null ? Path.of(".") : parent, target.getFileName() + ".", ".tmp");
    Bundle validated;
    try {
      write(temporary, actor, opponent, seedBase, captureGames);
      validated = load(temporary);
      publishNewFile(temporary, target);
    } finally {
      Files.deleteIfExists(temporary);
    }
    return validated;
  }

  /**
   * 永続化済み固定入力データを逐次処理復号し、形式・スキーマ・形状を検証する。
   *
   * @param file 固定入力データファイル
   * @return 複製を挟まず標準形式の連続バッファを直接所有する二役割の固定入力データ
   * @throws IOException 形式、スキーマまたは形状が一致しない場合
   */
  public static Bundle load(Path file) throws IOException {
    return load(file, DecisionInputSchema.fingerprint());
  }

  static Bundle load(Path file, String expectedSchemaFingerprint) throws IOException {
    try (InputStream fileInput = Files.newInputStream(file);
        DataInputStream input = new DataInputStream(new BufferedInputStream(fileInput))) {
      long magic = input.readLong();
      int formatVersion = input.readInt();
      if (magic != MAGIC || formatVersion != FORMAT_VERSION) {
        throw new IOException(
            "unsupported Decision replay corpus format: magic="
                + Long.toUnsignedString(magic, 16)
                + " version="
                + formatVersion);
      }
      String schemaFingerprint = readText(input);
      if (!schemaFingerprint.equals(expectedSchemaFingerprint)) {
        throw new IOException(
            "Decision replay corpus schema mismatch: file="
                + schemaFingerprint
                + " runtime="
                + expectedSchemaFingerprint);
      }
      long seedBase = input.readLong();
      int captureGames = input.readInt();
      if (captureGames < 1) {
        throw new IOException("invalid Decision replay corpus capture game count: " + captureGames);
      }
      if (input.readInt() != ROLE_COUNT) {
        throw new IOException("Decision replay corpus must contain exactly two roles");
      }
      EnumMap<Role, Corpus> corpora = new EnumMap<>(Role.class);
      for (int roleIndex = 0; roleIndex < ROLE_COUNT; roleIndex++) {
        Role role = parseRole(readText(input));
        if (corpora.put(role, readCorpus(input, role)) != null) {
          throw new IOException("duplicate Decision replay corpus role: " + role);
        }
      }
      if (input.read() != -1) {
        throw new IOException("trailing Decision replay corpus bytes");
      }
      Corpus actor = corpora.get(Role.ACTOR_POLICY_AND_VALUE);
      Corpus opponent = corpora.get(Role.OPPONENT_POLICY_ONLY);
      if (actor == null || opponent == null) {
        throw new IOException("Decision replay corpus is missing a required role");
      }
      return new Bundle(actor, opponent, formatVersion, schemaFingerprint, seedBase, captureGames);
    } catch (EOFException | IllegalArgumentException e) {
      throw new IOException("invalid Decision replay corpus: " + file, e);
    }
  }

  private static void write(
      Path file, Corpus actor, Corpus opponent, long seedBase, int captureGames)
      throws IOException {
    requireRole(actor, Role.ACTOR_POLICY_AND_VALUE);
    requireRole(opponent, Role.OPPONENT_POLICY_ONLY);
    if (captureGames < 1) {
      throw new IllegalArgumentException("invalid Decision replay corpus capture provenance");
    }
    try (OutputStream fileOutput = Files.newOutputStream(file);
        DataOutputStream output = new DataOutputStream(new BufferedOutputStream(fileOutput))) {
      output.writeLong(MAGIC);
      output.writeInt(FORMAT_VERSION);
      writeText(output, DecisionInputSchema.fingerprint());
      output.writeLong(seedBase);
      output.writeInt(captureGames);
      output.writeInt(ROLE_COUNT);
      writeCorpus(output, actor);
      writeCorpus(output, opponent);
    }
  }

  private static void writeCorpus(DataOutputStream output, Corpus corpus) throws IOException {
    Report report = corpus.report();
    writeText(output, corpus.role().name());
    writeText(output, report.sampling());
    output.writeInt(report.buckets().size());
    ByteBuffer scratch = ByteBuffer.allocate(0);
    for (Map.Entry<DecisionBucket, BucketStatistics> entry : report.buckets().entrySet()) {
      DecisionBucket bucket = entry.getKey();
      BucketStatistics statistics = entry.getValue();
      DecisionHostBatch batch = corpus.storedBatch(bucket);
      DecisionHostBatch.RowSlice rows = batch.sliceRows(0, batch.size());
      int categoricalElements = rows.inputCategoricalElementCount();
      int numericElements = rows.inputNumericElementCount();
      int categoricalBytes = Math.multiplyExact(categoricalElements, Short.BYTES);
      int numericBytes = Math.multiplyExact(numericElements, Float.BYTES);
      int capacity = Math.max(categoricalBytes, numericBytes);
      if (scratch.capacity() < capacity) {
        scratch = ByteBuffer.allocate(capacity);
      }
      output.writeInt(bucket.legalActionCapacity());
      output.writeInt(bucket.actionTransitionCapacity());
      output.writeLong(statistics.observedRows());
      output.writeInt(statistics.retainedRows());
      output.writeInt(statistics.targetRows());
      output.writeInt(categoricalElements);
      output.writeInt(numericElements);
      rows.copyInputCategoriesTo(scratch.asShortBuffer());
      output.write(scratch.array(), 0, categoricalBytes);
      rows.copyInputNumericsTo(scratch.asFloatBuffer());
      output.write(scratch.array(), 0, numericBytes);
    }
  }

  private static Corpus readCorpus(DataInputStream input, Role role) throws IOException {
    String sampling = readText(input);
    int bucketCount = input.readInt();
    if (bucketCount < 1 || bucketCount > MAXIMUM_BUCKETS) {
      throw new IOException("invalid Decision replay corpus bucket count: " + bucketCount);
    }
    LinkedHashMap<DecisionBucket, DecisionHostBatch> batches = new LinkedHashMap<>();
    LinkedHashMap<DecisionBucket, BucketStatistics> statistics = new LinkedHashMap<>();
    for (int bucketIndex = 0; bucketIndex < bucketCount; bucketIndex++) {
      DecisionBucket bucket = new DecisionBucket(input.readInt(), input.readInt());
      long observedRows = input.readLong();
      int rowCount = input.readInt();
      int targetRows = input.readInt();
      DecisionInputLayout layout = new DecisionInputLayout(rowCount, bucket);
      int categoricalElements = input.readInt();
      int numericElements = input.readInt();
      if (observedRows < rowCount
          || rowCount < 1
          || targetRows < rowCount
          || categoricalElements != layout.categoricalElementCount()
          || numericElements != layout.numericElementCount()) {
        throw new IOException("invalid Decision replay corpus bucket metadata: " + bucket);
      }
      short[] categories = new short[categoricalElements];
      float[] numerics = new float[numericElements];
      for (int index = 0; index < categories.length; index++) {
        categories[index] = input.readShort();
      }
      for (int index = 0; index < numerics.length; index++) {
        numerics[index] = Float.intBitsToFloat(input.readInt());
      }
      DecisionHostBatch batch =
          DecisionHostBatch.takeEncodedInferenceBatch(rowCount, bucket, categories, numerics);
      if (batches.put(bucket, batch) != null) {
        throw new IOException("duplicate Decision replay corpus bucket: " + bucket);
      }
      statistics.put(bucket, new BucketStatistics(observedRows, rowCount, targetRows));
    }
    return Corpus.restore(role, batches, sampling, statistics);
  }

  private static Role parseRole(String name) throws IOException {
    try {
      return Role.valueOf(name);
    } catch (IllegalArgumentException e) {
      throw new IOException("unknown Decision replay corpus role: " + name, e);
    }
  }

  private static void requireRole(Corpus corpus, Role expected) {
    if (corpus.role() != expected) {
      throw new IllegalArgumentException(
          "Decision replay corpus role mismatch: expected="
              + expected
              + " actual="
              + corpus.role());
    }
  }

  private static String readText(DataInputStream input) throws IOException {
    int length = input.readInt();
    if (length < 0 || length > MAXIMUM_TEXT_BYTES) {
      throw new IOException("invalid Decision replay corpus text length: " + length);
    }
    byte[] encoded = input.readNBytes(length);
    if (encoded.length != length) {
      throw new EOFException("truncated Decision replay corpus text");
    }
    return new String(encoded, StandardCharsets.UTF_8);
  }

  private static void writeText(DataOutputStream output, String value) throws IOException {
    byte[] encoded = value.getBytes(StandardCharsets.UTF_8);
    if (encoded.length > MAXIMUM_TEXT_BYTES) {
      throw new IOException("Decision replay corpus text is too long");
    }
    output.writeInt(encoded.length);
    output.write(encoded);
  }

  private static void publishNewFile(Path source, Path target) throws IOException {
    Files.createLink(target, source);
    Files.delete(source);
  }
}
