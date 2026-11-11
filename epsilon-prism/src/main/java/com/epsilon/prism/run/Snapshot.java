package com.epsilon.prism.run;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.settings.MemoryNetwork;
import com.epsilon.prism.settings.MemorySettings;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * ネットワークのバージョンのディレクトリ（系譜の snapshots/&lt;t&gt; と、refit の out/snapshot）。
 *
 * <pre>
 * m0/            打つネットワークの重み（weights.bin と model.json。architecture は epsilon-prism-v2）。方策の記憶があれば平均のネットワーク A
 * iterate/       学習中のネットワーク θ（方策の記憶があるときだけ。なければ m0 が θ）
 * average.json   方策の記憶の記録（{@link AverageRecord}。方策の記憶があるときだけ）
 * policy.json    バージョン番号（{@link PolicyParams}）
 * fit.json       refit のバージョンを作った学習の記録（{@link FitRecord}: 行・コホート・計画・結果。系譜のバージョンには無く、系譜の記録は iterations.jsonl にある）
 * </pre>
 *
 * 対局・評価・牌譜解析は m0 を読む（{@link #open}）。学習の続きは {@link #iterate} を読む。書くときは書きかけのディレクトリに全てを書いて
 * から rename するので、&lt;t&gt; があれば中身はそろっている。refit は &lt;t&gt;.tmp に書く。系譜の公開は &lt;t&gt;.part を中身ごと
 * fsync してから、行き先が無いときだけ rename する。
 */
public final class Snapshot {
  public static final String MODEL = "m0";
  public static final String POLICY = "policy.json";
  public static final String ITERATE = "iterate";
  public static final String AVERAGE = "average.json";
  public static final String FIT = "fit.json";

  /** 方策の記憶の記録を書く（値はいつも有限）。 */
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  /** 学習の記録を読み書きする（損失の NaN も書く）。 */
  private static final Gson FIT_GSON =
      new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

  /**
   * 方策の記憶の記録。
   *
   * @param scheme 決め方の名前（{@link MemorySettings#scheme}）
   * @param eta このバージョンで学習中のネットワークを混ぜた割合（1 なら平均のネットワーク = 学習中のネットワーク）
   * @param from 平均を始めたバージョン
   */
  public record AverageRecord(String scheme, double eta, int from) {}

  /**
   * バージョンを作った学習の記録。
   *
   * @param rows 行を読んだ run のディレクトリ（行は rows/&lt;t&gt;）
   * @param firstCohort 学んだ最初のコホート
   * @param lastCohort 学んだ最後のコホート
   * @param replicate アームの複製の番号（乱数の種を変える）
   * @param plan 学習の計画
   * @param result 学習の結果
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
   * 読んだバージョン（{@link #load}）。
   *
   * @param network 打つネットワーク
   * @param params バージョン番号
   * @param fit 学習の記録
   */
  public record Loaded(PrismNetwork network, PolicyParams params, FitRecord fit) {}

  private Snapshot() {}

  /** 打つネットワークのディレクトリ。 */
  public static Path model(Path snapshot) {
    return snapshot.resolve(MODEL);
  }

  /** 学習中のネットワークのディレクトリ（iterate/ があればそれ、なければ m0）。 */
  public static Path iterate(Path snapshot) {
    Path iterate = snapshot.resolve(ITERATE);
    return Files.isDirectory(iterate) ? iterate : model(snapshot);
  }

  /** バージョンのネットワーク（打つネットワーク m0 か、学習中のネットワーク）のディレクトリ。 */
  public static Path net(Path snapshot, MemoryNetwork network) {
    return switch (network) {
      case AVERAGE -> model(snapshot);
      case ITERATE -> iterate(snapshot);
    };
  }

  /**
   * バージョンの打つネットワーク m0 を推論用に開く。貪欲な手は s の argmax（{@link #open(Path, Path, List, NetworkRuntime, int,
   * Precision, InferencePath)}）。
   *
   * @param precision eager のバックエンドの計算の型
   * @param inference ネットワークの推論のバックエンド
   */
  public static NetworkPolicy open(
      Path snapshot,
      List<Device> devices,
      NetworkRuntime runtime,
      int maxBatch,
      Precision precision,
      InferencePath inference)
      throws IOException {
    return open(snapshot, model(snapshot), devices, runtime, maxBatch, precision, inference);
  }

  /**
   * バージョン {@code snapshot} のネットワーク {@code net}（{@link #model} か {@link #iterate}）を推論用に開く
   * （{@link NetworkPolicy}）。番号はバージョンのものを使う。
   *
   * @param precision eager のバックエンドの計算の型
   * @param inference ネットワークの推論のバックエンド
   */
  public static NetworkPolicy open(
      Path snapshot,
      Path net,
      List<Device> devices,
      NetworkRuntime runtime,
      int maxBatch,
      Precision precision,
      InferencePath inference)
      throws IOException {
    return NetworkPolicy.open(
        PolicyParams.read(snapshot.resolve(POLICY)), net, devices, runtime, maxBatch, precision, inference);
  }

  /**
   * 学習の記録つきのバージョンを読む。ネットワークの形式と、入力・ヘッドのフィンガープリントは {@link PrismNetwork#load} が照合する。
   *
   * @param snapshot バージョンのディレクトリ
   * @param manager ネットワークのパラメーターを持つマネージャー
   * @return 読んだバージョン
   * @throws IOException 読めないか、学習の記録がないとき
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

  /** バージョンを書く（m0 と policy.json）。&lt;t&gt;.tmp に全てを書いてから &lt;t&gt; へ rename する。rename の前に止まったバージョンの残りがあれば、消してから書く。 */
  public static void write(Path snapshot, PrismNetwork network, PolicyParams params)
      throws IOException {
    Path temporary = temporary(snapshot);
    fill(temporary, network, params);
    publish(temporary, snapshot);
  }

  /**
   * 学習の記録つきのバージョンを書く（m0・policy.json・fit.json。refit のバージョン）。書き方は {@link #write(Path, PrismNetwork,
   * PolicyParams)} と同じ。
   */
  public static void write(Path snapshot, PrismNetwork network, PolicyParams params, FitRecord fit)
      throws IOException {
    Path temporary = temporary(snapshot);
    fill(temporary, network, params);
    Files.writeString(temporary.resolve(FIT), FIT_GSON.toJson(fit));
    publish(temporary, snapshot);
  }

  /**
   * バージョンの中身（m0 と policy.json）を書きかけのディレクトリに書く。完成した名前への rename は呼び出し元が行う。
   *
   * @param part 中身の無い書きかけのディレクトリ（無ければ作る）
   */
  public static void fill(Path part, PrismNetwork network, PolicyParams params) throws IOException {
    Files.createDirectories(part);
    network.save(model(part));
    params.write(part.resolve(POLICY));
  }

  /**
   * 保存したネットワークのディレクトリ（weights.bin と model.json）を、そのまま書きかけのバージョンの m0 に写す。ネットワークを読み込まない
   * ので、重みはビット単位で同じ。
   *
   * @param part 中身の無い書きかけのディレクトリ（無ければ作る）
   * @param modelDirectory 保存したネットワークのディレクトリ
   * @throws IOException 保存したネットワークでないとき
   */
  public static void fillCopy(Path part, Path modelDirectory, PolicyParams params) throws IOException {
    requireSaved(modelDirectory);
    Files.createDirectories(part);
    copyModel(modelDirectory, model(part));
    params.write(part.resolve(POLICY));
  }

  /**
   * 方策の記憶つきのバージョンの中身を書く。学習中のネットワークは、保存したディレクトリをファイルのまま iterate/ に写す。平均の
   * ネットワークは m0 に、記録は average.json に書く。
   *
   * @param part 中身の無い書きかけのディレクトリ（無ければ作る）
   * @param iterateDirectory 学習中のネットワークを保存したディレクトリ
   * @throws IOException 保存したネットワークでないとき
   */
  public static void fillAveraged(
      Path part, Path iterateDirectory, PrismNetwork average, AverageRecord record, PolicyParams params)
      throws IOException {
    requireSaved(iterateDirectory);
    Files.createDirectories(part);
    copyModel(iterateDirectory, part.resolve(ITERATE));
    average.save(model(part));
    Files.writeString(part.resolve(AVERAGE), GSON.toJson(record));
    params.write(part.resolve(POLICY));
  }

  private static void requireSaved(Path directory) throws IOException {
    if (!Files.exists(directory.resolve("model.json"))) {
      throw new IOException("Not a saved Prism model: " + directory);
    }
  }

  /** 保存したネットワークのディレクトリのファイルを、そのまま写す（重みはビット単位で同じ）。 */
  static void copyModel(Path from, Path to) throws IOException {
    Files.createDirectories(to);
    try (Stream<Path> files = Files.list(from)) {
      for (Path file : files.toList())
        Files.copy(file, to.resolve(file.getFileName()), StandardCopyOption.COPY_ATTRIBUTES);
    }
  }

  /** 書きかけの &lt;t&gt;.tmp を作る（前の残りがあれば消して作り直す）。 */
  private static Path temporary(Path snapshot) throws IOException {
    Path temporary = snapshot.resolveSibling(snapshot.getFileName() + ".tmp");
    if (Files.exists(temporary)) deleteTree(temporary);
    Files.createDirectories(temporary);
    return temporary;
  }

  /** &lt;t&gt;.tmp を &lt;t&gt; に差し替える（再開したときに、同じバージョンを作り直すことがある）。 */
  private static void publish(Path temporary, Path snapshot) throws IOException {
    if (Files.exists(snapshot)) deleteTree(snapshot);
    Files.move(temporary, snapshot, StandardCopyOption.ATOMIC_MOVE);
  }

  /** ディレクトリを中身ごと消す。無ければ何もしない。 */
  public static void deleteTree(Path path) throws IOException {
    DurableFiles.delete(path);
  }
}
