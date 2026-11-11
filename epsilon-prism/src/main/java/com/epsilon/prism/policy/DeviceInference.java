package com.epsilon.prism.policy;

import ai.djl.Device;
import com.epsilon.agent.runtime.DecisionDevicePipeline;
import java.nio.FloatBuffer;

/**
 * 1 つのデバイスで動く推論のバックエンド（{@link InferencePath}）。ネットワークの重みをデバイスに置き、1 回の受付の s を出す。CPU では
 * 呼び出したスレッドで（{@link #score}）、GPU では共通のデバイスパイプラインのスロットで（{@link #work}）計算する。s はどちらも、受付の中の
 * バッチを順に並べ、各バッチの手の順の FP32 を詰めてつないだもの（{@link Encoding#answer} が読む並び）。
 */
sealed interface DeviceInference extends AutoCloseable permits EagerInference, HipInference {

  Device device();

  /**
   * 呼び出したスレッドで s を出す。スロットは使わない。
   *
   * @param encoding 符号化を終えた作業領域
   * @return s（呼び出し元が持つ）
   */
  FloatBuffer score(Encoding encoding);

  /**
   * スロットで行う 1 回の受付のジョブを作る。デバイスの処理はまだ積まない。
   *
   * @param encoding 符号化を終えた作業領域
   * @return ジョブ
   */
  SlotWork work(Encoding encoding);

  @Override
  void close();

  /**
   * スロットで行う 1 回の受付のジョブ（{@link DecisionDevicePipeline.Work} の、バックエンドごとの中身）。提出・完了・回収のスレッドへ順に
   * 渡る。
   */
  interface SlotWork extends AutoCloseable {

    /**
     * 入力の転送と順伝播をスロットに積み、積んだ全ての処理の後に {@code slot.computeReady()} を記録する。途中で投げるときも、積んだ処理の後に
     * 印を記録してから投げる（記録できなければ {@link #settle} が false を返す）。
     *
     * @param slot スロット
     */
    void submit(DecisionDevicePipeline.Slot slot);

    /**
     * 失敗したときに回収する。積んだ処理が終わるのを待つ。
     *
     * @param slot スロット
     * @param problem 失敗（待てなかったら、その理由を付け足す）
     * @return 積んだ処理が終わったのを確かめたら true（スロットを使い回せる）
     */
    boolean settle(DecisionDevicePipeline.Slot slot, Throwable problem);

    /**
     * s を返す。{@code slot.computeReady()} の後に、位置 0 から読む。
     *
     * @param slot スロット
     * @return s
     */
    FloatBuffer scores(DecisionDevicePipeline.Slot slot);

    /** 受付ごとのリソースを解放する。スロットの計算ストリームのスコープで呼ばれる。 */
    @Override
    void close();
  }
}
