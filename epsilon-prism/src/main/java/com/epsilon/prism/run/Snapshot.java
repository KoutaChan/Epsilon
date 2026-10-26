package com.epsilon.prism.run;

import ai.djl.Device;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.model.PrismModel;
import com.epsilon.prism.policy.NetPolicyService;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.policy.PriorSettings;
import com.epsilon.spi.PolicyExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * 公開した方策の版 snapshots/&lt;t&gt;。
 *
 * <pre>
 * m0/            打つ網の重み（weights.bin と model.json。architecture は epsilon-prism-v1）。方策の記憶があれば平均の網 A
 * iterate/       学習の網 θ（方策の記憶があるときだけ。なければ m0 が θ）
 * average.json   方策の記憶の記録（{@link AverageRecord}。方策の記憶があるときだけ）
 * policy.json    版の番号（{@link PolicyParams}）
 * </pre>
 *
 * 対局・評価・牌譜解析は m0 を読む。学習の続きは {@link #iterate} を読む。書くときは &lt;t&gt;.tmp に全てを書いてから rename するので、&lt;t&gt;
 * があれば中身はそろっている。
 */
public final class Snapshot {
  public static final String POLICY = "policy.json";
  public static final String ITERATE = "iterate";
  public static final String AVERAGE = "average.json";

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  /**
   * 方策の記憶の記録。
   *
   * @param scheme {@link Averaging#scheme}
   * @param eta この版で学習の網を混ぜた割合（1 なら平均の網 = 学習の網）
   * @param totalWeight power の重みの累計
   * @param from 平均を始めた版
   */
  public record AverageRecord(String scheme, double eta, double totalWeight, int from) {}

  private Snapshot() {}

  /** 打つ網のディレクトリ。 */
  public static Path model(Path snapshot) {
    return snapshot.resolve("m0");
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

  /** 版の番号。 */
  public static PolicyParams params(Path snapshot) throws IOException {
    return PolicyParams.read(snapshot.resolve(POLICY));
  }

  /** 学習用に網を読む（devices の先頭が正本で、残りはデータ並列の複製）。生徒と蒸留の温め込みに使う。 */
  public static PrismModel load(Path snapshot, NetworkDevices devices) throws IOException {
    return PrismModel.load(model(snapshot), devices, true);
  }

  /** 推論用に網を読む（凍結。認証・蒸留の教師・保留の指標の採点に使う）。 */
  public static PrismModel loadFrozen(Path snapshot, Device device) throws IOException {
    return PrismModel.load(model(snapshot), NetworkDevices.of(device), false);
  }

  /**
   * 推論用に開く（貪欲 = s の argmax）。版の番号を読み、網を各デバイスに読む。
   *
   * @param prior retain の問い合わせに書く p0 の係数
   */
  public static NetPolicyService open(
      Path snapshot,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      int multiBatch,
      PriorSettings prior)
      throws IOException {
    return open(snapshot, model(snapshot), devices, execution, maxBatch, multiBatch, prior);
  }

  /**
   * 版 snapshot の網 net（{@link #model} か {@link #iterate}）を推論用に開く。番号は版のもの。
   *
   * @param prior retain の問い合わせに書く p0 の係数
   */
  public static NetPolicyService open(
      Path snapshot,
      Path net,
      List<Device> devices,
      PolicyExecutionContext execution,
      int maxBatch,
      int multiBatch,
      PriorSettings prior)
      throws IOException {
    PolicyParams params = params(snapshot);
    List<PrismModel> copies = new ArrayList<>(devices.size());
    try {
      for (Device device : devices)
        copies.add(PrismModel.load(net, NetworkDevices.of(device), false));
      return new NetPolicyService(params, prior, copies, execution, maxBatch, multiBatch);
    } catch (IOException | RuntimeException | Error failure) {
      for (PrismModel copy : copies) copy.close();
      throw failure;
    }
  }

  /** 版を書く。&lt;t&gt;.tmp に全てを書いてから &lt;t&gt; へ rename する。rename の前に止まった版の残りがあれば消してから書く。 */
  public static void write(Path snapshot, PrismModel model, PolicyParams params)
      throws IOException {
    Path temporary = temporary(snapshot);
    model.save(model(temporary));
    params.write(temporary.resolve(POLICY));
    publish(temporary, snapshot);
  }

  /**
   * 版 from の網をそのまま版 to として書く（番号は params）。認証で蒸留した網を受け入れなかった反復の π_{t+1} = π_t に使う。網は読まずにファイルを
   * 写す。
   */
  public static void copy(Path from, Path to, PolicyParams params) throws IOException {
    writeFiles(model(from), to, params);
  }

  /**
   * 保存済みの網のディレクトリ（weights.bin と model.json。{@link PrismModel#save} の形）をそのまま版 to として書く（番号は params）。蒸留した網
   * （work/&lt;t&gt;/distilled）を受け入れた反復の公開に使う。網は読まずにファイルを写すので、重みはビット単位で同じ。
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
   * 方策の記憶つきの版を書く: 学習の網（保存済みの網のディレクトリ iterateDirectory をファイルのまま写す）を iterate/ に、平均の網 average を m0
   * に、記録を average.json に書く。
   */
  public static void writeAveraged(
      Path iterateDirectory,
      PrismModel average,
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

  /** 版の方策の記憶の記録（なければ null）。 */
  public static AverageRecord averageRecord(Path snapshot) throws IOException {
    Path path = snapshot.resolve(AVERAGE);
    return Files.exists(path) ? GSON.fromJson(Files.readString(path), AverageRecord.class) : null;
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
