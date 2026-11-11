package com.epsilon.classic.learn;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * 事前学習を中断して続きから学ぶための、途中の状態のディレクトリ（{@link Pretrainer#train}）。中には進み具合（progress.json）と、
 * 保存するたびに新しい番号で書くテンソルのファイル（tensors-NNNNN.bin）がある。テンソルのファイルには、今の重み、AdamW のモーメント、
 * 最良のエポックの重みを書く。テンソルのファイルを書き終えてから進み具合を書き替え、その後で古いテンソルのファイルを消す。そのため、
 * どこで落ちても、進み具合が指すファイルはそろっている。
 *
 * <p>テンソルのファイルは DataOutputStream（ビッグエンディアン）で書く。ただし FP32 の値は、リトルエンディアンでまとめて書く。
 *
 * <pre>
 * i32 パラメーターの数  i64 AdamW のステップ数  u8 最良の重みがあれば 1
 * パラメーターごと: UTF 名前  i32 値の数  f32 × 値の数 （今の重み・1 次のモーメント・2 次のモーメント・最良の重み（あれば）の順）
 * </pre>
 */
final class PretrainState {

  private static final String PROGRESS = "progress.json";
  private static final String TENSORS_PREFIX = "tensors-";
  private static final String TENSORS_SUFFIX = ".bin";
  private static final int BUFFER = 1 << 20;

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

  /**
   * 進み具合。
   *
   * @param settings 学習の設定（続きから学ぶときは、エポックの数のほかは同じであること）
   * @param architecture ネットワークの形の名前
   * @param trainRows 学習用の行の数
   * @param validationRows 検証用の行の数
   * @param baselines 検証用の行で測った基準（学び始める前に 1 度だけ測る）
   * @param epochs 終えたエポックの記録
   * @param best 最良のエポック（無ければ −1）
   * @param epoch 今のエポック
   * @param batch 今のエポックで次に学ぶバッチの番号
   * @param lossSum 今のエポックの、ステップごとの損失の和
   * @param epochSteps 今のエポックのステップ数
   * @param steps ステップの総数
   * @param segment テンソルのファイルの番号（保存するたびに 1 増やす）
   */
  record Progress(
      Pretrainer.Settings settings,
      String architecture,
      int trainRows,
      int validationRows,
      Pretrainer.Baselines baselines,
      List<Pretrainer.Epoch> epochs,
      int best,
      int epoch,
      int batch,
      double lossSum,
      long epochSteps,
      long steps,
      int segment) {}

  /**
   * テンソルの値。どの配列もパラメーターの順に並ぶ。
   *
   * @param names パラメーターの名前
   * @param adamSteps AdamW のステップ数
   * @param weights 今の重み
   * @param means AdamW の 1 次のモーメント
   * @param variances AdamW の 2 次のモーメント
   * @param best 最良のエポックの重み（無ければ null）
   */
  record Tensors(List<String> names, long adamSteps, float[][] weights, float[][] means, float[][] variances,
      float[][] best) {}

  private PretrainState() {}

  /** 進み具合を読む。まだ無ければ null。 */
  static Progress progress(Path directory) throws IOException {
    Path file = directory.resolve(PROGRESS);
    if (!Files.exists(file)) {
      return null;
    }
    return GSON.fromJson(Files.readString(file, StandardCharsets.UTF_8), Progress.class);
  }

  /** 進み具合が指すテンソルを読む。 */
  static Tensors tensors(Path directory, Progress progress) throws IOException {
    Path file = tensorsFile(directory, progress.segment());
    try (DataInputStream in =
        new DataInputStream(new BufferedInputStream(Files.newInputStream(file), BUFFER))) {
      int count = in.readInt();
      long adamSteps = in.readLong();
      boolean hasBest = in.readUnsignedByte() != 0;
      String[] names = new String[count];
      float[][] weights = new float[count][];
      float[][] means = new float[count][];
      float[][] variances = new float[count][];
      float[][] best = hasBest ? new float[count][] : null;
      for (int index = 0; index < count; index++) {
        names[index] = in.readUTF();
        int length = in.readInt();
        weights[index] = floats(in, length);
        means[index] = floats(in, length);
        variances[index] = floats(in, length);
        if (best != null) {
          best[index] = floats(in, length);
        }
      }
      if (in.read() != -1) {
        throw new IOException("Trailing bytes after the tensors of " + file);
      }
      return new Tensors(List.of(names), adamSteps, weights, means, variances, best);
    } catch (EOFException truncated) {
      throw new IOException("Truncated tensors " + file, truncated);
    }
  }

  /** テンソルを新しいファイルに書いてから進み具合を書き替え、最後に古いテンソルのファイルを消す。 */
  static void write(Path directory, Progress progress, Tensors tensors) throws IOException {
    Files.createDirectories(directory);
    Path file = tensorsFile(directory, progress.segment());
    Path temporary = file.resolveSibling(file.getFileName() + ".tmp");
    int count = tensors.names().size();
    try (DataOutputStream out =
        new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(temporary), BUFFER))) {
      out.writeInt(count);
      out.writeLong(tensors.adamSteps());
      out.writeByte(tensors.best() == null ? 0 : 1);
      for (int index = 0; index < count; index++) {
        out.writeUTF(tensors.names().get(index));
        int length = tensors.weights()[index].length;
        out.writeInt(length);
        write(out, tensors.weights()[index], length);
        write(out, tensors.means()[index], length);
        write(out, tensors.variances()[index], length);
        if (tensors.best() != null) {
          write(out, tensors.best()[index], length);
        }
      }
    } catch (IOException | RuntimeException failure) {
      Files.deleteIfExists(temporary);
      throw failure;
    }
    Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    Path progressFile = directory.resolve(PROGRESS);
    Path progressTemporary = progressFile.resolveSibling(PROGRESS + ".tmp");
    Files.writeString(progressTemporary, GSON.toJson(progress), StandardCharsets.UTF_8);
    Files.move(progressTemporary, progressFile, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    try (Stream<Path> listed = Files.list(directory)) {
      for (Path old : listed.toList()) {
        String name = old.getFileName().toString();
        if (name.startsWith(TENSORS_PREFIX) && name.endsWith(TENSORS_SUFFIX) && !old.equals(file)) {
          Files.delete(old);
        }
      }
    }
  }

  private static Path tensorsFile(Path directory, int segment) {
    return directory.resolve(String.format(Locale.ROOT, "%s%05d%s", TENSORS_PREFIX, segment, TENSORS_SUFFIX));
  }

  private static void write(DataOutputStream out, float[] values, int length) throws IOException {
    if (values.length != length) {
      throw new IllegalArgumentException("Tensor of " + values.length + " values where " + length + " are expected");
    }
    ByteBuffer bytes = ByteBuffer.allocate(Float.BYTES * length).order(ByteOrder.LITTLE_ENDIAN);
    bytes.asFloatBuffer().put(values);
    out.write(bytes.array());
  }

  private static float[] floats(DataInputStream in, int length) throws IOException {
    if (length < 0) {
      throw new IOException("Negative tensor length " + length);
    }
    byte[] raw = new byte[Float.BYTES * length];
    in.readFully(raw);
    float[] values = new float[length];
    ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
    return values;
  }
}
