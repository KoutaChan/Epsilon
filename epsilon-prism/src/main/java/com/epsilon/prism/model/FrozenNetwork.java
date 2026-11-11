package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.TrainingKernels;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * 保存したネットワークを 1 つのデバイスに読み込み、重みを変えずに s を出す（推論だけ）。順伝播は、読み込むときに選んだ計算の型で行う
 * （{@link Precision}）。方策を eager で推論するバックエンドと、系譜のアンカー（学習中のネットワーク θ_t を凍結したものが出す s）
 * が使う。
 */
public final class FrozenNetwork implements AutoCloseable {

  private final NDManager manager;
  private final PrismNetwork network;
  private final Precision precision;

  private FrozenNetwork(NDManager manager, PrismNetwork network, Precision precision) {
    this.manager = manager;
    this.network = network;
    this.precision = precision;
  }

  /**
   * 保存したネットワークを読み込む。
   *
   * @param model ネットワークのディレクトリ（weights.bin と model.json）
   * @param device デバイス
   * @param precision 順伝播の計算の型
   * @return 読み込んだネットワーク
   * @throws IOException 読めないとき、または形式・フィンガープリント・重みが合わないとき
   */
  public static FrozenNetwork load(Path model, Device device, Precision precision) throws IOException {
    Objects.requireNonNull(precision, "precision");
    // エンジンを初めて取る前に、全てのモデルに共通の実行時の既定値（PyTorch の CPU スレッド数）を決める
    NetworkRuntime.initialize();
    NDManager manager = NDManager.newBaseManager(device, PtEngine.ENGINE_NAME);
    try {
      return new FrozenNetwork(manager, PrismNetwork.load(model, manager), precision);
    } catch (IOException | RuntimeException | Error failure) {
      manager.close();
      throw failure;
    }
  }

  public Device device() {
    return manager.getDevice();
  }

  public PrismShape shape() {
    return network.shape();
  }

  /** 1 回の処理で作る配列を付けるマネージャーを作る。 */
  public NDManager newScope() {
    return manager.newSubManager();
  }

  /**
   * 順伝播のインデックスをデバイスへ送るバッファを作る。バッファはネットワークのマネージャーに付くので、ネットワークを閉じると解放される。
   * 1 つのスレッドが順に使う。
   *
   * @return バッファ
   */
  public PinnedUploads newUploads() {
    return PinnedUploads.on(manager);
  }

  /** 自動微分を記録しない区間を始める。順伝播と出力の整形はこの中で行う。 */
  public InferenceMode inference() {
    return manager.getEngine().newInferenceMode();
  }

  /**
   * 1 つのバッチの s を出す。順伝播で作る配列は、入力のマネージャーに付く。
   *
   * @param inputs バッチの入力
   * @return s [行, 手の容量]（FP32。列 k は行の k 番目の手）
   */
  public NDArray scores(NetworkInputs inputs) {
    return scores(inputs, null);
  }

  /**
   * 1 つのバッチの s を出す。バックボーンの Attention と正規化には、学習のレーンと同じ自前の演算を記録なしで使う。順伝播で作る配列は、
   * 入力のマネージャーに付く。
   *
   * @param inputs バッチの入力
   * @param attention バックボーンの Attention と正規化に使う自前の演算（null なら djl-rocm の演算）
   * @return s [行, 手の容量]（FP32。列 k は行の k 番目の手）
   */
  public NDArray scores(NetworkInputs inputs, TrainingKernels attention) {
    Pass pass =
        Pass.inference(precision, attention == null ? null : AttentionTape.forwardOnly(attention));
    return network.forward(new ParameterStore(manager, false), inputs, pass).scores();
  }

  @Override
  public void close() {
    manager.close();
  }
}
