package com.epsilon.prism.run;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.policy.PriorSettings;
import com.epsilon.spi.PolicyExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * 網の版のディレクトリ（系譜の snapshots/&lt;t&gt; と refit の out/snapshot）。
 *
 * <pre>
 * m0/            打つ網の重み（weights.bin と model.json。architecture は epsilon-prism-v2）。方策の記憶があれば平均の網 A
 * iterate/       学習の網 θ（方策の記憶があるときだけ。なければ m0 が θ）
 * average.json   方策の記憶の記録（{@link AverageRecord}。方策の記憶があるときだけ）
 * policy.json    版の番号（{@link PolicyParams}）
 * fit.json       refit の版を作った当てはめの記録（{@link FitRecord}: 行・コホート・計画・結果。系譜の版は持たず、記録は iterations.jsonl）
 * </pre>
 *
 * 対局・評価・牌譜解析は m0 を読む（{@link #open}）。学習の続きは {@link #iterate} を読む。書くときは &lt;t&gt;.tmp に全てを書いてから
 * rename するので、&lt;t&gt; があれば中身はそろっている。
 */
public final class Snapshot {
  public static final String MODEL = "m0";
  public static final String POLICY = "policy.json";
  public static final String ITERATE = "iterate";
  public static final String AVERAGE = "average.json";
  public static final String FIT = "fit.json";

  /** 方策の記憶の記録を書く（数は有限）。 */
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  /** 当てはめの記録を読み書きする（損失の NaN も書く）。 */
  private static final Gson FIT_GSON =
      new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

  /**
   * 方策の記憶の記録。
   *
   * @param scheme {@link Averaging#scheme}
   * @param eta この版で学習の網を混ぜた割合（1 なら平均の網 = 学習の網）
   * @param totalWeight power の重みの累計
   * @param from 平均を始めた版
   */
  public record AverageRecord(String scheme, double eta, double totalWeight, int from) {}

  /**
   * 版を作った当てはめの記録。
   *
   * @param rows 行を読んだ run のディレクトリ（行は rows-v2/&lt;t&gt;）
   * @param firstCohort 学んだ最初のコホート
   * @param lastCohort 学んだ最後のコホート
   * @param replicate 腕の複製の番号（種を変える）
   * @param plan 当てはめの計画
   * @param result 当てはめの結果
   */
  public record FitRecord(
      String rows, int firstCohort, int lastCohort, int replicate, FitPlan plan, FitResult result) {

    /**
     * 記録を確かめる。
     *
     * @throws IllegalArgumentException コホートの範囲か複製の番号が正しくないとき
     */
    public FitRecord {
      Objects.requireNonNull(rows, "rows");
      Objects.requireNonNull(plan, "plan");
      Objects.requireNonNull(result, "result");
      if (firstCohort < 1 || lastCohort < firstCohort || replicate < 0) {
        throw new IllegalArgumentException(
            "invalid cohorts " + firstCohort + ".." + lastCohort + " or replicate " + replicate);
      }
    }
  }

  /**
   * 読んだ版（{@link #load}）。
   *
   * @param network 打つ網
   * @param params 版の番号
   * @param fit 当てはめの記録
   */
  public record Loaded(PrismNetwork network, PolicyParams params, FitRecord fit) {}

  private Snapshot() {}

  /** 打つ網のディレクトリ。 */
  public static Path model(Path snapshot) {
    return snapshot.resolve(MODEL);
  }

  /** 学習の網のディレクトリ（iterate/ があればそれ、なければ m0）。 */
  public static Path iterate(Path snapshot) {
    Path iterate = snapshot.resolve(ITERATE);
    return Files.isDirectory(iterate) ? iterate : model(snapshot);
  }

  /** 網の名前（{@link Averaging#AVERAGE} = 打つ網 m0、{@link Averaging#ITERATE} = 学習の網）のディレクトリ。 */
  public static Path net(Path snapshot, String name) {
    return switch (name) {
      case Averaging.AVERAGE -> model(snapshot);
      case Averaging.ITERATE -> iterate(snapshot);
      default -> throw new IllegalArgumentException("Unknown net " + name);
    };
  }

  /**
   * 版の打つ網 m0 を推論用に開く（貪欲 = s の argmax。{@link #open(Path, Path, List, PolicyExecutionContext, int, PriorSettings,
   * InferencePath)}）。
   *
   * @param prior retain と s の最大が同点の問い合わせに書く p0 の係数
   * @param inference 網の推論の道（{@link PrismConfig#inference}）
   */
  public static PolicyService open(
      Path snapshot,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      PriorSettings prior,
      InferencePath inference)
      throws IOException {
    return open(snapshot, model(snapshot), devices, execution, maxBatch, prior, inference);
  }

  /**
   * 版 snapshot の網 net（{@link #model} か {@link #iterate}）を推論用に開く（{@link NetworkPolicy}）。番号は版のもの。
   *
   * @param prior retain と s の最大が同点の問い合わせに書く p0 の係数
   * @param inference 網の推論の道（{@link PrismConfig#inference}）
   */
  public static PolicyService open(
      Path snapshot,
      Path net,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      PriorSettings prior,
      InferencePath inference)
      throws IOException {
    return NetworkPolicy.open(
        PolicyParams.read(snapshot.resolve(POLICY)), net, devices, execution, maxBatch, prior, inference);
  }

  /**
   * 当てはめの記録つきの版を読む。網の形式と入力・頭の指紋は {@link PrismNetwork#load} が照らす。
   *
   * @param snapshot 版のディレクトリ
   * @param manager 網のパラメーターの管理元
   * @return 読んだ版
   * @throws IOException 読めないか、当てはめの記録がないとき
   */
  public static Loaded load(Path snapshot, NDManager manager) throws IOException {
    PolicyParams params = PolicyParams.read(snapshot.resolve(POLICY));
    FitRecord fit;
    try {
      fit = FIT_GSON.fromJson(Files.readString(snapshot.resolve(FIT)), FitRecord.class);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid PRISM fit record " + snapshot, invalid);
    }
    if (fit == null) {
      throw new IOException("Empty PRISM fit record " + snapshot);
    }
    return new Loaded(PrismNetwork.load(model(snapshot), manager), params, fit);
  }

  /** 版を書く（m0 と policy.json）。&lt;t&gt;.tmp に全てを書いてから &lt;t&gt; へ rename する。rename の前に止まった版の残りがあれば消してから書く。 */
  public static void write(Path snapshot, PrismNetwork network, PolicyParams params)
      throws IOException {
    Path temporary = temporary(snapshot);
    network.save(model(temporary));
    params.write(temporary.resolve(POLICY));
    publish(temporary, snapshot);
  }

  /**
   * 当てはめの記録つきの版を書く（m0・policy.json・fit.json。refit の版）。書き方は {@link #write(Path, PrismNetwork,
   * PolicyParams)} と同じ。
   */
  public static void write(Path snapshot, PrismNetwork network, PolicyParams params, FitRecord fit)
      throws IOException {
    Path temporary = temporary(snapshot);
    network.save(model(temporary));
    params.write(temporary.resolve(POLICY));
    Files.writeString(temporary.resolve(FIT), FIT_GSON.toJson(fit));
    publish(temporary, snapshot);
  }

  /** 網を directory.tmp に書いてから directory に差し替える（途中で止まっても読める網が残る。反復の途中の網の保存）。 */
  public static void save(PrismNetwork network, Path directory) throws IOException {
    Path temporary = directory.resolveSibling(directory.getFileName() + ".tmp");
    if (Files.exists(temporary)) deleteTree(temporary);
    network.save(temporary);
    if (Files.exists(directory)) deleteTree(directory);
    Files.move(temporary, directory, StandardCopyOption.ATOMIC_MOVE);
  }

  /**
   * 保存済みの網のディレクトリ（weights.bin と model.json）をそのまま版 to として書く（番号は params）。方策の記憶のない反復の公開に使う。網は
   * 読まずにファイルを写すので、重みはビット単位で同じ。
   */
  public static void writeFiles(Path modelDirectory, Path to, PolicyParams params)
      throws IOException {
    if (!Files.exists(modelDirectory.resolve("model.json")))
      throw new IOException("Not a saved Prism model: " + modelDirectory);
    Path temporary = temporary(to);
    copyModel(modelDirectory, model(temporary));
    params.write(temporary.resolve(POLICY));
    publish(temporary, to);
  }

  /**
   * 方策の記憶つきの版を書く: 学習の網（保存済みの網のディレクトリ iterateDirectory をファイルのまま写す）を iterate/ に、平均の網 average を
   * m0 に、記録を average.json に書く。
   */
  public static void writeAveraged(
      Path iterateDirectory,
      PrismNetwork average,
      AverageRecord record,
      Path to,
      PolicyParams params)
      throws IOException {
    if (!Files.exists(iterateDirectory.resolve("model.json")))
      throw new IOException("Not a saved Prism model: " + iterateDirectory);
    Path temporary = temporary(to);
    copyModel(iterateDirectory, temporary.resolve(ITERATE));
    average.save(model(temporary));
    Files.writeString(temporary.resolve(AVERAGE), GSON.toJson(record));
    params.write(temporary.resolve(POLICY));
    publish(temporary, to);
  }

  private static void copyModel(Path from, Path to) throws IOException {
    Files.createDirectories(to);
    try (Stream<Path> files = Files.list(from)) {
      for (Path file : files.toList())
        Files.copy(file, to.resolve(file.getFileName()), StandardCopyOption.COPY_ATTRIBUTES);
    }
  }

  /** 書きかけの &lt;t&gt;.tmp（残りがあれば消して作り直す）。 */
  private static Path temporary(Path snapshot) throws IOException {
    Path temporary = snapshot.resolveSibling(snapshot.getFileName() + ".tmp");
    if (Files.exists(temporary)) deleteTree(temporary);
    Files.createDirectories(temporary);
    return temporary;
  }

  /** &lt;t&gt;.tmp を &lt;t&gt; に差し替える（再開で同じ版を作り直すことがある）。 */
  private static void publish(Path temporary, Path snapshot) throws IOException {
    if (Files.exists(snapshot)) deleteTree(snapshot);
    Files.move(temporary, snapshot, StandardCopyOption.ATOMIC_MOVE);
  }

  /** ディレクトリを中身ごと消す。 */
  public static void deleteTree(Path path) throws IOException {
    try (Stream<Path> files = Files.walk(path)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);
    }
  }
}
