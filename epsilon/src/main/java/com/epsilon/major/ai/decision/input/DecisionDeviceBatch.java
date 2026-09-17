package com.epsilon.major.ai.decision.input;

import ai.djl.ndarray.NDArray;

/**
 * デバイスへ転送した Decision の入力と、学習時に付与する教師値を保持する。
 *
 * <p>入力の意味別ビューは{@link DecisionNetworkInputs}へ集約する。この型が直接所有する二本の連続バッファはビューの寿命を保つためのデバイス
 * 記憶領域であり、モデルは連続バッファの並びやオフセットを解釈しない。存在するインデックスは用途別ビューを保持し、学習教師値も同様に二本の連続バッファと
 * 用途別ビューを保持する。全テンソルの寿命は転送先管理元に従う。
 *
 * @param categoricalSlab カテゴリ値入力を連続配置したデバイス側記憶領域
 * @param numericSlab 数値入力を連続配置したデバイス側記憶領域
 * @param inputs モデルへ渡す名前付き論理ビュー
 * @param playerMemoryPresentIndices プレイヤーごとの履歴表現の存在トークンを指す行優先インデックス
 * @param transitionPresentIndices 存在する行動遷移を指す行優先インデックス
 * @param trainingTargets 学習時だけ存在する教師値記憶領域とビュー
 */
public record DecisionDeviceBatch(
    NDArray categoricalSlab,
    NDArray numericSlab,
    DecisionNetworkInputs inputs,
    NDArray playerMemoryPresentIndices,
    NDArray transitionPresentIndices,
    TrainingTargets trainingTargets) {

  /**
   * デバイス側バッチの行数を返す。
   *
   * @return バッチ行数
   */
  public int rowCount() {
    return inputs.rowCount();
  }

  /**
   * 行動・遷移軸の密な形状を返す。
   *
   * @return このバッチの容量区分
   */
  public DecisionBucket bucket() {
    return inputs.bucket();
  }

  /**
   * 行動 IDのパディングから合法候補マスクを導出する。
   *
   * @return 形状 {@code [rows, actionCapacity]} のboolean マスク
   */
  public NDArray legalActionMask() {
    return inputs.legalActionMask();
  }

  /**
   * 学習教師値を伴うバッチかを返す。
   *
   * @return 学習バッチなら {@code true}
   */
  public boolean hasTrainingTargets() {
    return trainingTargets != null;
  }

  /**
   * 転送済み学習教師値と用途別ビュー。
   *
   * <p>探索適用後の方策は実際の無作為抽出分布、対局生成方策は探索適用前の標準形式の分布であり、両者を交換してはならない。
   *
   * @param categoricalSlab 整数教師値を連続配置したデバイス側記憶領域
   * @param numericSlab 浮動小数教師値を連続配置したデバイス側記憶領域
   * @param chosenSlot 動的合法候補列で選択した格納位置
   * @param behaviorPolicy 探索適用後の合法候補確率
   * @param rolloutPolicy 探索適用前のモデル合法候補確率
   * @param valueTarget 期待効用のスカラー教師
   * @param advantage 効用の選択行動アドバンテージ
   * @param actorWeight 方策損失対象行の重み
   * @param sampleWeight 価値損失対象行の重み
   */
  public record TrainingTargets(
      NDArray categoricalSlab,
      NDArray numericSlab,
      NDArray chosenSlot,
      NDArray behaviorPolicy,
      NDArray rolloutPolicy,
      NDArray valueTarget,
      NDArray advantage,
      NDArray actorWeight,
      NDArray sampleWeight) {}
}
