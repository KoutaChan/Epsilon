package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.training.ParameterStore;
import com.epsilon.network.NetworkRuntime;
import com.epsilon.prism.hip.RelationAttention;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 保存した網を 1 つのデバイスに読んで、重みを変えずに s を出す（推論だけ）。GPU は BF16 の丸め、CPU は FP32 で順伝播する
 * （{@link Precision}）。方策の eager の推論の道と、系譜の錨（凍結した学習の網 θ_t の s）が使う。
 */
public final class FrozenNetwork implements AutoCloseable {

  private final NDManager manager;
  private final PrismNetwork network;
  private final Precision precision;

  private FrozenNetwork(NDManager manager, PrismNetwork network) {
    this.manager = manager;
    this.network = network;
    precision = manager.getDevice().isGpu() ? Precision.BFLOAT16 : Precision.FLOAT32;
  }

  /**
   * 保存した網を読む。
   *
   * @param model 網のディレクトリ（weights.bin と model.json）
   * @param device デバイス
   * @return 網
   * @throws IOException 読めないか、形式・指紋・重みが合わないとき
   */
  public static FrozenNetwork load(Path model, Device device) throws IOException {
    // エンジンを初めて取る前に、全モデル共通の実行既定値（PyTorch の CPU スレッド数）を決める
    NetworkRuntime.initialize();
    NDManager manager = NDManager.newBaseManager(device, PtEngine.ENGINE_NAME);
    try {
      return new FrozenNetwork(manager, PrismNetwork.load(model, manager));
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

  /** 1 回の仕事の配列を付ける管理元。 */
  public NDManager newScope() {
    return manager.newSubManager();
  }

  /**
   * 順伝播の索引をデバイスへ送る置き場を作る（網の管理元に付け、網を閉じると返る）。1 つのスレッドが順に使う。
   *
   * @return 置き場
   */
  public Uploads newUploads() {
    return Uploads.on(manager);
  }

  /** 自動微分を記録しない区間（順伝播と出力の整形をこの中で行う）。 */
  public InferenceMode inference() {
    return manager.getEngine().newInferenceMode();
  }

  /**
   * 1 つの塊の s を出す。順伝播の配列は入力の管理元に付く。
   *
   * @param inputs 塊の入力
   * @return s [行, 手の容量]（FP32。列 k は行の k 番目の手）
   */
  public NDArray scores(NetworkInputs inputs) {
    return scores(inputs, null);
  }

  /**
   * 1 つの塊の s を、背骨の注意を自前の演算（記録なし。学習のレーンと同じ演算）にして出す。順伝播の配列は入力の管理元に付く。
   *
   * @param inputs 塊の入力
   * @param attention 背骨の注意の自前の演算（null なら djl-rocm の演算）
   * @return s [行, 手の容量]（FP32。列 k は行の k 番目の手）
   */
  public NDArray scores(NetworkInputs inputs, RelationAttention attention) {
    Pass pass =
        Pass.inference(precision, attention == null ? null : AttentionTape.forwardOnly(attention));
    return network.forward(new ParameterStore(manager, false), inputs, pass).scores();
  }

  @Override
  public void close() {
    manager.close();
  }
}
