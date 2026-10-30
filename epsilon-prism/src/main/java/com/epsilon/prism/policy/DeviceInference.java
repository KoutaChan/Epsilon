package com.epsilon.prism.policy;

import ai.djl.Device;
import com.epsilon.runtime.DecisionDevicePipeline;
import java.nio.FloatBuffer;

/**
 * 1 つのデバイスの推論の道（{@link InferencePath}）。網の重みをデバイスに持ち、受け付けの s を、CPU では呼び出しスレッドで（{@link #score}）、
 * GPU では共通のデバイスパイプラインの実行枠で（{@link #work}）出す。s はどちらも、塊の順に塊の手の順の FP32 を詰めてつないだもの
 * （{@link Encoding#answer} が読む並び）。
 */
sealed interface DeviceInference extends AutoCloseable permits EagerInference, HipInference {

  Device device();

  /**
   * 呼び出しスレッドで s を出す（実行枠を使わない口）。
   *
   * @param encoding 符号化を終えた作業域
   * @return s（呼び手が持つ）
   */
  FloatBuffer score(Encoding encoding);

  /**
   * 実行枠の 1 回の受け付けの仕事を作る（デバイスの仕事はまだ積まない）。
   *
   * @param encoding 符号化を終えた作業域
   * @return 仕事
   */
  SlotWork work(Encoding encoding);

  @Override
  void close();

  /** 実行枠の 1 回の受け付けの仕事（{@link DecisionDevicePipeline.Work} の道ごとの中身。提出・完了・回収のスレッドへ順に渡る）。 */
  interface SlotWork extends AutoCloseable {

    /**
     * 入力の転送と順伝播を実行枠に積み、全ての仕事の後に {@code slot.computeReady()} を記録する。途中で投げるときも、積んだ仕事の後に印を残して
     * から投げる（残せなければ {@link #settle} が false を返す）。
     *
     * @param slot 実行枠
     */
    void submit(DecisionDevicePipeline.Slot slot);

    /**
     * 失敗の回収: 積んだ仕事が終わるのを待つ。
     *
     * @param slot 実行枠
     * @param problem 失敗（待てない理由を付け足す）
     * @return 積んだ仕事の終わりを確かめたら true（実行枠を使い回せる）
     */
    boolean settle(DecisionDevicePipeline.Slot slot, Throwable problem);

    /**
     * s（{@code slot.computeReady()} の後に、位置 0 から読む）。
     *
     * @param slot 実行枠
     * @return s
     */
    FloatBuffer scores(DecisionDevicePipeline.Slot slot);

    /** 受け付けごとの資源を返す（実行枠の計算ストリームのスコープで呼ばれる）。 */
    @Override
    void close();
  }
}
