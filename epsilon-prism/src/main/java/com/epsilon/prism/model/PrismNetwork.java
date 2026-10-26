package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.ParameterList;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.nano.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.nano.ai.decision.input.DecisionGroupedInferenceBatch;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.nano.ai.decision.policy.TransitionPooling;
import com.epsilon.nano.ai.model.DecisionInferenceExecution;
import com.epsilon.nano.ai.model.EpsilonDecisionFeatureSpine;
import com.epsilon.nano.ai.model.EpsilonDecisionFrozenFeatures;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;

/**
 * 生徒 π_θ（論文 3.4 M3、3.7）。合法手ごとのスコア s(o, a) だけを出し、値の頭は持たない。s は効用の単位で読み、打つときは argmax、学習では
 * s/λ_c をロジットにする。盤面と候補手の表現は Nano と同じ形を使う。
 *
 * <p>出力の 1 単位の層（output）は自動混合精度でも FP32 で計算する（{@link #head}）。BF16 の出力では値が 0.002〜0.004 刻みに丸まり、近い手が
 * 同点になる。隠れ層（candidateHidden）は BF16 のまま。打つときの融合推論（{@link #inference}）も同じ s を FP32 で出す。
 *
 * <p>子ブロックの登録順は stateEncoder、stateReadout、候補の表現、candidateHidden、output。DJL は子ブロック名に登録順の番号を付けるので、
 * 順を変えると保存した版の重みの名前が合わなくなる。output は白紙の網ではゼロで初期化する（出力 0）。
 */
final class PrismNetwork extends AbstractBlock {
  static final String ARCHITECTURE = "epsilon-prism-v1";
  private final ModelConfig config;
  private final EpsilonDecisionFeatureSpine featureSpine;
  private final Linear candidateHidden, output;

  PrismNetwork(ModelConfig config) {
    this.config = config;
    int hidden = config.hiddenSize();
    featureSpine =
        new EpsilonDecisionFeatureSpine(hidden, TransitionPooling.CANDIDATE_REPRESENTATION);
    addChildBlock("stateEncoder", featureSpine.stateEncoder());
    addChildBlock("stateReadout", featureSpine.stateReadout());
    featureSpine.candidates().registerWith(this::addChildBlock);
    candidateHidden = addChildBlock("candidateHidden", Linear.builder().setUnits(hidden).build());
    output = addChildBlock("output", Linear.builder().setUnits(1).build());
    output.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0));
    output.getDirectParameters().get("bias").setInitializer(new ConstantInitializer(0));
  }

  EpsilonDecisionFrozenFeatures frozenFeatures(NDManager owner, DataType computeType) {
    return featureSpine.newFrozenInference(owner, computeType);
  }

  /** 学習と採点の出力 s。[行, 合法手の容量]、FP32（合法でない手の値は使わない）。 */
  NDArray scores(ParameterStore store, DecisionDeviceBatch batch, boolean training) {
    return scores(store, batch, training, new PairList<>());
  }

  /**
   * 学習と採点の出力 s。[行, 合法手の容量]、FP32。
   *
   * @param parameters 実行時パラメータ（{@code EpsilonTileRelationEncoder#createRuntimeParameters}。空なら順伝播のたびに作る）
   */
  NDArray scores(
      ParameterStore store,
      DecisionDeviceBatch batch,
      boolean training,
      PairList<String, Object> parameters) {
    var state =
        featureSpine.encodeState(
            store, batch.inputs(), batch.playerMemoryPresentIndices(), training, parameters);
    NDArray candidate =
        featureSpine
            .candidates()
            .encodeRepresentation(
                store,
                state,
                batch.inputs(),
                batch.transitionPresentIndices(),
                training,
                parameters);
    return scoreCandidates(
        store, candidate, batch.bucket().legalActionCapacity(), training, parameters);
  }

  /**
   * 打つときの出力 s（FP32）。全盤面・遷移を共有し、合法候補の s だけを元の行動順で返す。同じ網と盤面なら {@link #scores} と同じ s。
   */
  NDArray inference(
      ParameterStore store,
      DecisionGroupedInferenceBatch batch,
      EpsilonDecisionFrozenFeatures frozen,
      DecisionInferenceExecution.Forward forward) {
    var parameters = new PairList<String, Object>();
    var state =
        frozen == null
            ? featureSpine.encodeState(
                store,
                batch.stateCategories(),
                batch.stateNumerics(),
                batch.playerMemoryPresentIndices(),
                false,
                parameters)
            : frozen.encodeState(
                store,
                batch.stateCategories(),
                batch.stateNumerics(),
                batch.playerMemoryPresentIndices(),
                forward,
                parameters);
    var encoder = frozen == null ? featureSpine.candidates() : frozen.candidates();
    NDArray candidates =
        encoder.encodeGroupedInferenceRepresentation(
            store, state, batch, forward == null ? null : forward.candidates(), parameters);
    return scoreCandidates(store, candidates, 1, false, parameters).reshape(-1);
  }

  private NDArray scoreCandidates(
      ParameterStore store,
      NDArray candidate,
      int actionCapacity,
      boolean training,
      PairList<String, Object> parameters) {
    NDArray hidden =
        candidateHidden.forward(store, new NDList(candidate), training, parameters).getFirst();
    return head(store, output, hidden, training)
        .reshape(candidate.getShape().get(0), actionCapacity);
  }

  /**
   * 1 単位の出力層 y = Σ silu(h)·w + b を FP32 で計算する（最後の軸を消す）。Linear の行列積は自動混合精度で BF16 に下がり出力も BF16 に
   * 丸まるので、要素ごとの積と和で計算する（autocast は要素ごとの積を下げず、和の入力はすでに FP32）。silu(h) は BF16 のまま FP32 の重みと掛け、
   * 型昇格で FP32 にする。重みは ParameterStore から読むので、勾配・名前・登録・保存の形式は Linear のまま。
   *
   * <p>推論の ParameterStore は重みを写さず、パラメーターの配列そのもの（管理元はモデル）を返す。そこから reshape などで配列を作るとモデルの
   * 管理元に付いて閉じられずに残るので、重みは形を変えずにブロードキャストで使い、新しい配列は入力の管理元に付ける。
   */
  private static NDArray head(
      ParameterStore store, Linear layer, NDArray hidden, boolean training) {
    NDArray input = EpsilonMahjongStateEncoder.silu(hidden);
    NDManager working = input.getManager();
    Device device = input.getDevice();
    ParameterList parameters = layer.getDirectParameters();
    NDArray weight =
        float32(store.getValue(parameters.get("weight"), device, training), working);
    NDArray bias = float32(store.getValue(parameters.get("bias"), device, training), working);
    // weight は [1, 幅] なので [行, 容量, 幅] にもそのまま広がる。bias は [1]
    return input.mul(weight).sum(new int[] {input.getShape().dimension() - 1}).add(bias);
  }

  /** パラメーターの FP32 の見方。FP32 ならそのまま返し（写さない）、変換したときだけ新しい配列を作業用の管理元に付ける。 */
  private static NDArray float32(NDArray parameter, NDManager working) {
    if (parameter.getDataType() == DataType.FLOAT32) return parameter;
    NDArray cast = parameter.getNDArrayInternal().differentiableCast(DataType.FLOAT32);
    cast.attach(working);
    return cast;
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType type, Shape... ignored) {
    int hidden = config.hiddenSize();
    featureSpine
        .stateEncoder()
        .initialize(
            manager,
            type,
            DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_CATEGORIES),
            DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_NUMERICS));
    featureSpine
        .stateReadout()
        .initialize(
            manager, type, new Shape(-1, EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT, hidden));
    featureSpine.candidates().initialize(manager, type);
    candidateHidden.initialize(manager, type, new Shape(-1, -1, hidden));
    output.initialize(manager, type, new Shape(-1, -1, hidden));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore store, NDList inputs, boolean training, PairList<String, Object> parameters) {
    throw new UnsupportedOperationException("Use the typed scores entrypoint");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] shapes) {
    return new Shape[] {new Shape(-1, -1)};
  }
}
