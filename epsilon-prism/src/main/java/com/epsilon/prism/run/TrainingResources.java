package com.epsilon.prism.run;

import ai.djl.Device;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.prism.grp.ActivePhi;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiResolver;
import com.epsilon.prism.grp.PhiSettings;
import com.epsilon.prism.grp.PhiSetup;
import com.epsilon.prism.grp.PhiValue;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 系譜の JVM が反復をまたいで 1 つずつ持つ、収集・学習・評価のリソース。符号化と GPU で実行するリソース、学習器のストリーム、卓の
 * コア、Φ（反復ごとに GRP の項目とモードを決める {@link PhiResolver}）を持つ。Φ は純 Java で、局末を作った進行のスレッドがその場で
 * 呼ぶ。閉じる順は開いた順の逆。
 */
public final class TrainingResources implements AutoCloseable {

  /** 収集と評価を実行するリソース（符号化のプールと GPU の推論のパイプライン）。 */
  private final NetworkRuntime runtime;

  /**
   * 学習器にストリームを貸すリソース（{@link com.epsilon.prism.learn.Fit#prepare}）。PyTorch のキャッシュはストリームごとにあるので、
   * JVM が動いている間は同じストリームで学習し、前の反復の作業領域のキャッシュを使い回す。反復ごとに作り直すと、使われないキャッシュが
   * 反復ごとに 1 回分ずつ GPU に残る。GPU のストリームは、開くときに評価と収集より先に取る。ストリームは最後に返されたものから貸し直される
   * ので、最初の学習のときに取ると、収集が返したばかりのスロットのストリームをキャッシュごと持っていってしまう。そうなると、次の評価と
   * 収集は新しいストリームでキャッシュを作り直すことになる。
   */
  private final NetworkRuntime learning;

  /** 収集と評価の卓のコア。 */
  private final TableLoop table;

  /** 反復ごとの Φ を決める。 */
  private final PhiResolver resolver;

  private TrainingResources(NetworkRuntime runtime, NetworkRuntime learning, TableLoop table, PhiResolver resolver) {
    this.runtime = runtime;
    this.learning = learning;
    this.table = table;
    this.resolver = resolver;
  }

  /**
   * run の設定からリソースを開く。途中で失敗したら、開いた分を閉じる。
   *
   * @param root run のディレクトリ（phi.jsonl と mode.jsonl を置く）
   * @throws IOException phi.jsonl か mode.jsonl が読めないか書けないとき
   */
  public static TrainingResources open(PrismSettings settings, Path root) throws IOException {
    InferenceSettings inference = settings.inference();
    var runtime = new NetworkRuntime(inference.encoders(), inference.slots(), inference.readyBatches());
    var learning = NetworkRuntime.forLearning();
    TableLoop table = null;
    PhiResolver resolver = null;
    try {
      for (Device device : settings.learner().networkDevices()) {
        if (device.isGpu()) {
          learning.streams(device);
        }
      }
      table = new TableLoop(settings.table());
      resolver =
          PhiResolver.open(
              new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed()), new GrpFiles(root));
      return new TrainingResources(runtime, learning, table, resolver);
    } catch (IOException | RuntimeException | Error failure) {
      try (runtime;
          learning;
          TableLoop openedTable = table;
          PhiResolver openedResolver = resolver) {
        // 開いた分を逆の順に閉じる
      } catch (RuntimeException | Error closing) {
        failure.addSuppressed(closing);
      }
      throw failure;
    }
  }

  /** 収集と評価を実行するリソース。 */
  public NetworkRuntime runtime() {
    return runtime;
  }

  /** 学習器にストリームを貸すリソース（反復をまたいで同じもの）。 */
  public NetworkRuntime learning() {
    return learning;
  }

  /** 収集と評価の卓のコア。 */
  public TableLoop table() {
    return table;
  }

  /**
   * 反復 t の局末の値 Φ を決める（{@link PhiResolver#resolve}）。
   *
   * @param iteration 反復に固定した設定の行の、Φ のセクション（固定の GRP は、この行のチェックポイントと q で開く）
   * @throws IOException 既に使っている項目か、固定の GRP が開けないとき
   */
  public ActivePhi phi(int t, PhiSettings iteration) throws IOException {
    return resolver.resolve(t, iteration);
  }

  /**
   * 記録した値のとおりの Φ を開く（Φ の段の完了マーカーに書いた値で集め直すとき）。最後に開いた値と同じなら、それを使う。
   *
   * @throws IOException 値のチェックポイント・表・q のどれかが読めないか、sha が合わないとき
   */
  public ActivePhi phi(PhiValue value) throws IOException {
    return resolver.reopen(value);
  }

  /** Φ・卓のコア・学習器のストリーム・実行のリソースの順に閉じる（どれかが失敗しても残りを閉じる）。 */
  @Override
  public void close() {
    try (runtime;
         learning;
         table;
         resolver) {
      // 宣言の逆の順に閉じる
    }
  }
}
