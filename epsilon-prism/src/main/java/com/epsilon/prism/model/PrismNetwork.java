package com.epsilon.prism.model;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.hip.RelationCodes;
import com.epsilon.prism.input.FrontierField;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * PRISM の網。関係の偏りつきの背骨 → 遷移ごとの交差注意 → 価値グラフの頭で s を出し、学習では補助の頭も計算する。
 *
 * <p>子の登録の順は保存の形式の一部（DJL は子の名前に登録の番号を付ける）: stateEmbedding、encoder0〜encoder{L−1}、finalNorm、
 * candidate、discardValue、gates、reading、outcome。補助の頭（reading・outcome）は最後に置き、推論の重みの束
 * （{@link #inferenceParameters()}）に入れない。
 *
 * <p>前線の腕（{@link PrismShape#frontier()}）は候補層だけが違い、入力に前線の区画を求める。model.json には前線の入力の指紋
 * （{@link FrontierField#fingerprint()}）も書く。
 *
 * <p>白紙では価値グラフの最後の層が全て 0 なので s ≡ 0（PASS と詰め物も 0）。
 */
public final class PrismNetwork extends AbstractBlock {

  /** model.json の網の形式。 */
  public static final String ARCHITECTURE = "epsilon-prism-v2";

  private static final String WEIGHTS = "weights.bin";
  private static final String METADATA = "model.json";
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  /** model.json の中身（frontierSchema は前線の腕だけ、ほかは null で書かない）。 */
  private record Metadata(
      String architecture,
      String inputSchema,
      String frontierSchema,
      String headSchema,
      PrismShape shape) {}

  private final PrismShape shape;
  private final TokenEmbedding embedding;
  private final EncoderLayer[] encoders;
  private final RmsNorm finalNorm;
  private final CandidateLayer candidate;
  private final DiscardValue discardValue;
  private final ValueGraphHead gates;
  private final ReadingHeads reading;
  private final OutcomeHeads outcome;

  /**
   * 網を組む（パラメーターは {@link #initialize(NDManager)} で作る）。
   *
   * @param shape 形
   */
  public PrismNetwork(PrismShape shape) {
    this.shape = shape;
    embedding = addChildBlock("stateEmbedding", new TokenEmbedding(shape));
    encoders = new EncoderLayer[shape.depth()];
    for (int layer = 0; layer < encoders.length; layer++) {
      encoders[layer] = addChildBlock("encoder" + layer, new EncoderLayer(shape, layer));
    }
    finalNorm = addChildBlock("finalNorm", new RmsNorm(shape));
    candidate = addChildBlock("candidate", new CandidateLayer(shape));
    discardValue = addChildBlock("discardValue", new DiscardValue(shape));
    gates = addChildBlock("gates", new ValueGraphHead(shape));
    reading = addChildBlock("reading", new ReadingHeads(shape));
    outcome = addChildBlock("outcome", new OutcomeHeads(shape));
  }

  public PrismShape shape() {
    return shape;
  }

  /**
   * パラメーターを FP32 で作る（初期値は管理元のエンジンの乱数の種で決まる）。
   *
   * @param manager パラメーターの管理元
   */
  public void initialize(NDManager manager) {
    initialize(manager, DataType.FLOAT32);
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... ignored) {
    for (Block child : getChildren().values()) {
      child.initialize(manager, dataType);
    }
  }

  /**
   * 順伝播。autocast は開かずに呼ぶ（丸める所は {@link Precision} が決める）。段の区切りに順伝播の種類の時計の印を付ける
   * （{@link FitProbe}）。
   *
   * @param store パラメーターの値の取り元
   * @param inputs 入力（作った配列は入力の管理元に付く）
   * @param pass 順伝播の種類
   * @return 出力
   * @throws IllegalArgumentException 入力の前線の区画の有無が網の前線の腕と合わないとき
   */
  public Outputs forward(ParameterStore store, NetworkInputs inputs, Pass pass) {
    if (shape.frontier() != (inputs.frontier() != null)) {
      throw new IllegalArgumentException(
          shape.frontier()
              ? "the frontier arm needs frontier inputs"
              : "frontier inputs given to a network without the frontier arm");
    }
    FitProbe probe = pass.probe();
    Context context =
        new Context(
            store, inputs.manager(), pass.precision(), pass.training(), pass.tape(), probe);
    BatchIndex index = new BatchIndex(inputs, pass);
    probe.mark(Stage.BATCH_INDEX);
    NDArray states = backbone(context, inputs, index);
    NDArray decision = states.get(":, {}, :", TokenType.ROUND.first());
    NDArray transitions = candidate.forward(context, inputs, index, states);
    probe.mark(Stage.CANDIDATE);
    NDArray values =
        discardValue.forward(context, transitions, inputs.get(Tensor.DISCARD_EXACT));
    NDArray representations = context.toFloat(transitions);
    NDArray scores =
        gates.scores(
            context, index, representations, values, decision, inputs.get(Tensor.ACTION_EXACT));
    probe.mark(Stage.VALUE_GRAPH);
    Outputs.Auxiliary auxiliary =
        pass.auxiliary() == null
            ? null
            : auxiliary(context, index, representations, values, decision, states);
    probe.mark(Stage.AUX_HEADS);
    return new Outputs(scores, index.legal, index.rowActions, index.discardValueSlots, auxiliary);
  }

  /** 背骨: 埋め込み → L 層 → 最後の正規化。局・席・牌の [行, 39, H] を返す。 */
  private NDArray backbone(Context context, NetworkInputs inputs, BatchIndex index) {
    FitProbe probe = context.probe();
    int rows = inputs.rows();
    long hidden = shape.hidden();
    NDList parts = new NDList();
    for (TokenType type : TokenType.values()) {
      if (type == TokenType.EVENT) {
        if (index.maxEvents > 0) {
          NDArray events = embedding.embed(context, inputs, index, type);
          parts.add(
              NDArrays.scatterRows(events, index.eventSlots, (long) rows * index.maxEvents)
                  .reshape(rows, index.maxEvents, hidden));
        }
      } else {
        parts.add(
            embedding.embed(context, inputs, index, type).reshape(rows, type.perRow(), hidden));
      }
    }
    NDArray tokens = NDArrays.concat(parts, 1);
    probe.mark(Stage.EMBED);
    NDArray codes = relationCodes(context, inputs, index);
    probe.mark(Stage.RELATION);
    for (int layer = 0; layer < encoders.length; layer++) {
      int queries = layer == encoders.length - 1 ? TokenType.FIXED_TOKENS : index.tokens;
      tokens = encoders[layer].forward(context, tokens, index.tokenMask, codes, queries);
    }
    NDArray states = finalNorm.apply(context, tokens);
    probe.mark(Stage.FINAL_NORM);
    return splitBackward(context, states);
  }

  /**
   * 計測のときは背骨の出力を葉にする（{@link FitProbe#cuts}）。損失の逆伝播は頭の側（{@link Stage#HEADS_BACKWARD}）で止まり、
   * 最後の正規化と最後の層の O 射影・FFN は、葉に溜めた頭からの勾配の注入として別に計れる（{@link AttentionTape}）。注入する勾配は頭からの
   * 寄与の和そのものなので、勾配は葉にしないときと同じ式。
   */
  private NDArray splitBackward(Context context, NDArray states) {
    AttentionTape tape = context.tape();
    return tape == null || !context.probe().cuts()
        ? states
        : tape.cut(states, Stage.O_FFN_BACKWARD, encoders.length - 1);
  }

  /**
   * 関係の符号 [行, S, S, 2]（INT16）。GPU の学習で注意の記録の演算が符号も作るなら自前の演算（{@link AttentionTape#relationCodes}）、
   * ほかは eager（CPU・djl-rocm の注意・推論）。どちらも同じビット。
   */
  private static NDArray relationCodes(Context context, NetworkInputs inputs, BatchIndex index) {
    RelationCodes kernel = context.tape() == null ? null : context.tape().relationCodes();
    return kernel == null ? RelationClass.codes(inputs, index) : RelationClass.codes(kernel, inputs, index);
  }

  /** 補助の頭（候補の結果と読み）。 */
  private Outputs.Auxiliary auxiliary(
      Context context,
      BatchIndex index,
      NDArray representations,
      NDArray values,
      NDArray decision,
      NDArray states) {
    NDArray candidates = null;
    if (index.candidateCount > 0) {
      NDArray chosen =
          ValueGraphHead.aggregate(
              representations,
              values,
              index.candidateTransitions,
              index.candidateMask,
              index.candidateCount,
              index.candidateWidth);
      candidates =
          outcome.forward(context, chosen, NDArrays.gatherRows(decision, index.candidateRows));
    }
    NDArray readings =
        index.readingCount == 0
            ? null
            : reading.forward(context, NDArrays.gatherRows(states, index.readingRows));
    return new Outputs.Auxiliary(candidates, readings);
  }

  /**
   * 推論の重みの束（HIP の WeightLayout の元）。登録の順で、補助の頭を除く。
   *
   * @return 名前とパラメーター
   */
  public PairList<String, Parameter> inferenceParameters() {
    Set<String> auxiliary = auxiliaryChildren();
    PairList<String, Parameter> result = new PairList<>();
    for (Pair<String, Parameter> pair : getParameters()) {
      String child = pair.getKey().substring(0, pair.getKey().indexOf('_'));
      if (!auxiliary.contains(child)) {
        result.add(pair);
      }
    }
    return result;
  }

  /**
   * この網の重みを、同じ形の網 other の重みと混ぜる（方策の記憶: w ← (1 − eta)·w + eta·w_other。{@link ParameterMix#mix}）。
   *
   * @param other 混ぜる網
   * @param eta 混ぜる割合（(0, 1]。1 なら other の写し）
   * @throws IllegalArgumentException eta が範囲の外か、パラメーターの名前か形が違うとき
   */
  public void mixIn(PrismNetwork other, double eta) {
    ParameterMix.mix(getParameters(), other.getParameters(), eta);
  }

  /**
   * 全てのパラメーターの軸の役を返す。
   *
   * @return 軸の役の表
   */
  public WidthPlan widthPlan() {
    Map<String, List<Axis>> axes = new LinkedHashMap<>();
    for (Pair<String, Block> child : getChildren()) {
      Module module = (Module) child.getValue();
      for (Pair<String, Parameter> pair : module.getParameters()) {
        axes.put(child.getKey() + "_" + pair.getKey(), module.axes(pair.getValue()));
      }
    }
    return new WidthPlan(axes);
  }

  /**
   * 重み（weights.bin）と model.json（網の形式・入力の指紋・前線の腕なら前線の入力の指紋・頭の指紋・形）を書く。
   *
   * @param directory 書く先
   * @throws IOException 書けないとき
   */
  public void save(Path directory) throws IOException {
    Files.createDirectories(directory);
    try (DataOutputStream out =
        new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(directory.resolve(WEIGHTS))))) {
      saveParameters(out);
    }
    Files.writeString(
        directory.resolve(METADATA),
        GSON.toJson(
            new Metadata(
                ARCHITECTURE,
                InputSchema.fingerprint(),
                frontierSchema(shape),
                TargetSchema.fingerprint(),
                shape)));
  }

  /**
   * 保存した網の形を読む。網の形式と指紋（入力・前線の腕なら前線の入力・頭）を照らす。
   *
   * @param directory 保存した所
   * @return 形
   * @throws IOException 読めないか、形式・指紋が合わないとき
   */
  public static PrismShape readShape(Path directory) throws IOException {
    Metadata metadata;
    try {
      metadata = GSON.fromJson(Files.readString(directory.resolve(METADATA)), Metadata.class);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid PRISM metadata " + directory, invalid);
    }
    if (metadata == null || metadata.shape() == null || !ARCHITECTURE.equals(metadata.architecture())) {
      throw new IOException("Not a PRISM model: " + directory);
    }
    if (!InputSchema.fingerprint().equals(metadata.inputSchema())) {
      throw new IOException("PRISM input schema mismatch " + directory);
    }
    if (!TargetSchema.fingerprint().equals(metadata.headSchema())) {
      throw new IOException("PRISM head schema mismatch " + directory);
    }
    PrismShape shape =
        new PrismShape(
            metadata.shape().hidden(),
            metadata.shape().ffn(),
            metadata.shape().gate(),
            metadata.shape().depth(),
            metadata.shape().frontier());
    if (!Objects.equals(frontierSchema(shape), metadata.frontierSchema())) {
      throw new IOException("PRISM frontier schema mismatch " + directory);
    }
    return shape;
  }

  /** 前線の腕なら前線の入力の指紋、ほかは null。 */
  private static String frontierSchema(PrismShape shape) {
    return shape.frontier() ? FrontierField.fingerprint() : null;
  }

  /**
   * 保存した網を読む。
   *
   * @param directory 保存した所
   * @param manager パラメーターの管理元
   * @return 網
   * @throws IOException 読めないか、形式・指紋・重みが合わないとき
   */
  public static PrismNetwork load(Path directory, NDManager manager) throws IOException {
    PrismNetwork network = new PrismNetwork(readShape(directory));
    network.initialize(manager);
    try (DataInputStream in =
        new DataInputStream(new BufferedInputStream(Files.newInputStream(directory.resolve(WEIGHTS))))) {
      network.loadParameters(manager, in);
      if (in.read() != -1) {
        throw new IOException("Trailing PRISM weight data " + directory);
      }
    } catch (MalformedModelException invalid) {
      throw new IOException("Invalid PRISM weights " + directory, invalid);
    }
    return network;
  }

  private Set<String> auxiliaryChildren() {
    Set<String> names = new HashSet<>();
    for (Pair<String, Block> child : getChildren()) {
      if (child.getValue() == reading || child.getValue() == outcome) {
        names.add(child.getKey());
      }
    }
    return names;
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore store, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use forward(ParameterStore, NetworkInputs, Pass)");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    throw new UnsupportedOperationException("Shapes follow PrismShape");
  }
}
