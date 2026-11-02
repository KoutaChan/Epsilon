package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.prism.collect.GrpPhi;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/**
 * 候補の境目の点検: 作業が候補を公開するときに、固定の局末（約 {@value #SAMPLES} 個）とその候補の Φ を保存し、学習の JVM が切り替える反復の
 * 始めに同じ局末を計算して、全て有限で {@value #TOLERANCE} 以内で一致することを確かめる（作業と学習の JVM はスレッド数が違うのでビットでは比べ
 * ない）。
 *
 * <p>ファイルの形: 魔法の数・局末の数、局末ごとに公開系列の長さと float の列・次局の整数（局・本場・供託・4 人の点数）・4 席の Φ。
 *
 * @param inputs 見本の局末
 * @param utilities 作業が計算した 4 席の Φ
 */
public record BoundaryCheck(List<HandEndInput> inputs, double[][] utilities) {

  /** 見本の局末の数。 */
  public static final int SAMPLES = 4096;

  /** 一致とみなす Φ の差の上限。 */
  public static final double TOLERANCE = 1e-5;

  private static final int MAGIC = 0x50484942; // "PHIB"

  /** 学習の JVM の GrpPhi で計算するときの 1 バッチの上限（値は順伝播の行の数 maxBatch だけで決まり、これに依らない）。 */
  private static final int BATCH = 256;

  /** 学習の JVM で計算し終えるまでの上限。 */
  private static final long TIMEOUT_MINUTES = 10;

  public BoundaryCheck {
    inputs = List.copyOf(inputs);
    if (inputs.size() != utilities.length)
      throw new IllegalArgumentException("Boundary inputs and Φ values differ in length");
  }

  /**
   * 半荘の列から見本を決めて候補で計算する（種で決まる）。
   *
   * @throws IOException 候補が読めないとき
   */
  public static BoundaryCheck sample(List<RecordedGame> games, HandEndPhi candidate, long seed)
      throws IOException {
    List<HandEndInput> inputs = inputs(games, seed);
    return new BoundaryCheck(inputs, candidate.values(inputs).utilities());
  }

  /** 半荘の列の半荘が続く局末から、種で決まる最大 {@value #SAMPLES} 個（半荘と局末の順）。 */
  public static List<HandEndInput> inputs(List<RecordedGame> games, long seed) {
    int total = 0;
    for (RecordedGame game : games) total += game.continuingHandEnds();
    int count = Math.min(SAMPLES, total);
    int[] order = new int[total];
    for (int i = 0; i < total; i++) order[i] = i;
    var random = new SplittableRandom(seed);
    for (int i = 0; i < count; i++) {
      int j = i + random.nextInt(total - i);
      int swap = order[i];
      order[i] = order[j];
      order[j] = swap;
    }
    int[] chosen = Arrays.copyOf(order, count);
    Arrays.sort(chosen);
    List<HandEndInput> inputs = new ArrayList<>(count);
    int index = 0, next = 0;
    for (RecordedGame game : games) {
      for (int k = 0; k < game.continuingHandEnds() && next < count; k++, index++)
        if (chosen[next] == index) {
          inputs.add(game.input(k));
          next++;
        }
      if (next == count) break;
    }
    return inputs;
  }

  /** file に書く（一時ファイルに書いて fsync し、rename で差し替える）。 */
  public void write(Path file) throws IOException {
    Files.createDirectories(file.getParent());
    Path temporary = file.resolveSibling(file.getFileName() + ".tmp");
    try (var stream = new FileOutputStream(temporary.toFile());
        var out = new DataOutputStream(new BufferedOutputStream(stream))) {
      out.writeInt(MAGIC);
      out.writeInt(inputs.size());
      for (int i = 0; i < inputs.size(); i++) {
        HandEndInput input = inputs.get(i);
        out.writeInt(input.prefix().length);
        for (float value : input.prefix()) out.writeFloat(value);
        HandStart next = input.next();
        out.writeInt(next.kyokuIndex());
        out.writeInt(next.honba());
        out.writeInt(next.kyotaku());
        for (int score : next.scores()) out.writeInt(score);
        for (int seat = 0; seat < 4; seat++) out.writeDouble(utilities[i][seat]);
      }
      out.flush();
      stream.getFD().sync();
    }
    Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
  }

  /** file を読む。 */
  public static BoundaryCheck read(Path file) throws IOException {
    try (var in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
      if (in.readInt() != MAGIC) throw new IOException("Not a boundary check file: " + file);
      int count = in.readInt();
      if (count < 0 || count > SAMPLES) throw new IOException("Bad boundary sample count " + count + " in " + file);
      List<HandEndInput> inputs = new ArrayList<>(count);
      double[][] utilities = new double[count][4];
      for (int i = 0; i < count; i++) {
        int length = in.readInt();
        if (length <= 0 || length > 1 << 16) throw new IOException("Bad prefix length " + length + " in " + file);
        float[] prefix = new float[length];
        for (int j = 0; j < length; j++) prefix[j] = in.readFloat();
        int kyoku = in.readInt(), honba = in.readInt(), kyotaku = in.readInt();
        int[] scores = new int[4];
        for (int seat = 0; seat < 4; seat++) scores[seat] = in.readInt();
        inputs.add(new HandEndInput(prefix, new HandStart(kyoku, honba, kyotaku, scores)));
        for (int seat = 0; seat < 4; seat++) utilities[i][seat] = in.readDouble();
      }
      if (in.read() != -1) throw new IOException("Trailing bytes in " + file);
      return new BoundaryCheck(inputs, utilities);
    }
  }

  /**
   * 候補を開いたもの（{@link HandEndPhi#of}）が見本と合うか。
   *
   * @return 全て有限で {@value #TOLERANCE} 以内なら true
   * @throws IOException 候補が読めないとき
   */
  public boolean matches(PhiEntry candidate, EpsilonUtilityProfile profile) throws IOException {
    try (HandEndPhi phi = HandEndPhi.of(candidate, profile)) {
      return matches(phi.values(inputs).utilities());
    }
  }

  /**
   * 学習の JVM が開いた自己対局の GRP の推論（温度も表も重ねない）を、ラベルと同じ {@link GrpPhi} の経路（順伝播の行の数は推論の maxBatch）で
   * 計算して見本と合うか。GrpPhi が値を出せなければ（有限でないなど）false。
   *
   * @throws IOException 決まった時間で計算し終わらないとき
   */
  public boolean matches(EpsilonGrpInference inference, EpsilonUtilityProfile profile) throws IOException {
    int n = inputs.size();
    double[][] values = new double[n][];
    CountDownLatch done = new CountDownLatch(n);
    AtomicReference<Throwable> failure = new AtomicReference<>();
    ExecutorService driver =
        Executors.newSingleThreadExecutor(
            task -> {
              Thread thread = new Thread(task, "prism-grp-boundary");
              thread.setDaemon(true);
              return thread;
            });
    Consumer<Runnable> guarded =
        task ->
            driver.execute(
                () -> {
                  try {
                    task.run();
                  } catch (RuntimeException problem) {
                    failure.compareAndSet(null, problem);
                    while (done.getCount() > 0) done.countDown();
                  }
                });
    try (var phi = new GrpPhi(inference, profile, guarded, BATCH, 2 * Math.max(1, n))) {
      guarded.accept(
          () -> {
            for (int i = 0; i < n; i++) {
              int row = i;
              HandEndInput input = inputs.get(i);
              phi.request(
                  input.prefix(),
                  input.next().scores(),
                  result -> {
                    values[row] = result;
                    done.countDown();
                  });
            }
          });
      if (!done.await(TIMEOUT_MINUTES, TimeUnit.MINUTES))
        throw new IOException("The boundary check did not finish in " + TIMEOUT_MINUTES + " minutes");
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted during the boundary check", interrupted);
    } finally {
      driver.shutdownNow();
    }
    if (failure.get() != null) return false;
    return matches(values);
  }

  /** 計算した Φ が全て有限で、見本と {@value #TOLERANCE} 以内なら true。 */
  boolean matches(double[][] values) {
    if (values.length != utilities.length) return false;
    for (int i = 0; i < values.length; i++) {
      if (values[i] == null || values[i].length != 4) return false;
      for (int seat = 0; seat < 4; seat++) {
        double value = values[i][seat], expected = utilities[i][seat];
        if (!Double.isFinite(value) || !Double.isFinite(expected) || Math.abs(value - expected) > TOLERANCE)
          return false;
      }
    }
    return true;
  }
}
