package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.hip.HipLayer;
import com.epsilon.prism.hip.LayerRole;
import com.epsilon.prism.hip.RelationCodes;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.variant.PrismVariant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * PRISM のネットワーク。関係のバイアス付きのバックボーン → 遷移ごとの Cross-Attention → 価値グラフのヘッドの順に通して s を出す。
 * 学習では補助のヘッドも計算する。
 *
 * <p>子ブロックを登録する順は保存形式の一部（DJL は子の名前に登録の番号を付ける）: stateEmbedding、encoder0〜encoder{L−1}、finalNorm、
 * candidate、（候補層の 2 段目を持つ形だけ）reread、discardValue、gates、reading、outcome。補助のヘッド（reading・outcome）は最後に置き、
 * 推論で使う重み（{@link #inferenceParameters()}）には入れない。
 *
 * <p>層の種類（全てのトークンを読むか、局・席・牌だけを更新するか）と FFN の幅は形が決める（{@link PrismShape#layers()}）。型ごとの FFN を
 * 持つ形では、バックボーンの層 1 から後と最後の正規化が、局・席と牌で別々の重みを持つ。候補層の 2 段目（{@link CandidateReread}）は、層 0 の
 * 出力のイベントを読む。
 *
 * <p>ネットワークは 1 つのルールの入力の形式（{@link InputFormat}）を読む。トークンの数・入力の列の並び・キーのコード・ゲートの種類は
 * 形式が決める。形式は model.json の入力のフィンガープリントで決まる。読むときは、classpath にあるルールのモジュールの形式から、
 * フィンガープリントの合うものを選ぶ。
 *
 * <p>計算の型が {@link Precision#BFLOAT16} のときは、Linear・RMSNorm の出力・残差・SiLU・Attention の出力を BF16 に 1 回ずつ丸める。
 * Attention の中（q·k・バイアス・softmax・p·v）・候補の Cross-Attention・集約・ヘッドは FP32 で計算する。HIP も同じ所で丸める。
 * {@link Precision#FLOAT32} では全てを FP32 で計算する（CPU の Reviewer とテスト）。
 *
 * <p>初期化したままのネットワークでは価値グラフの最後の層が全て 0 なので、s ≡ 0 になる（PASS とパディングも 0）。
 */
public final class PrismNetwork extends AbstractBlock {

  /** model.json に書くネットワークの形式。 */
  public static final String ARCHITECTURE = "epsilon-prism-v2";

  /** model.json の中身。 */
  private record Metadata(String architecture, String inputSchema, String headSchema, PrismShape shape) {}

  /** 保存したネットワークの入力の形式と形。 */
  private record Header(InputFormat format, PrismShape shape) {}

  private final InputFormat format;
  private final PrismShape shape;

  /** バックボーンの層の種類（{@link PrismShape#layers()} の順）。 */
  private final LayerRole[] roles;

  /** 後の局・席・牌の層のキーとバリュー、または候補層の 2 段目がイベントを読むなら true。 */
  private final boolean keepEvents;
  private final TokenEmbedding embedding;
  private final EncoderLayer[] encoders;
  private final RmsNorm finalNorm;
  private final CandidateLayer candidate;

  /** 候補層の 2 段目（持たない形では null）。 */
  private final CandidateReread reread;
  private final DiscardValue discardValue;
  private final ValueGraphHead gates;
  private final ReadingHeads reading;
  private final OutcomeHeads outcome;

  /**
   * ネットワークを組む（パラメーターは {@link #initialize(NDManager)} で作る）。
   *
   * @param format 読む入力の形式
   * @param shape 形
   */
  public PrismNetwork(InputFormat format, PrismShape shape) {
    this.format = format;
    this.shape = shape;
    roles = shape.layers().stream().map(HipLayer::role).toArray(LayerRole[]::new);
    embedding = addChildBlock("stateEmbedding", new TokenEmbedding(shape, format));
    encoders = new EncoderLayer[shape.depth()];
    for (int layer = 0; layer < encoders.length; layer++) {
      encoders[layer] = addChildBlock("encoder" + layer, new EncoderLayer(shape, layer, format));
    }
    finalNorm = addChildBlock("finalNorm", new RmsNorm(shape));
    candidate = addChildBlock("candidate", new CandidateLayer(shape, format));
    reread = shape.reread() ? addChildBlock("reread", new CandidateReread(shape, format)) : null;
    long fixedLayers = Arrays.stream(roles).filter(role -> role == LayerRole.FIXED_TOKENS).count();
    keepEvents = fixedLayers > 1 || reread != null;
    discardValue = addChildBlock("discardValue", new DiscardValue(shape, format));
    gates = addChildBlock("gates", new ValueGraphHead(shape, format));
    reading = addChildBlock("reading", new ReadingHeads(shape, format));
    outcome = addChildBlock("outcome", new OutcomeHeads(shape, TargetSchema.of(format.variant())));
  }

  /** 読む入力の形式。 */
  public InputFormat format() {
    return format;
  }

  public PrismShape shape() {
    return shape;
  }

  /**
   * パラメーターを FP32 で作る（初期値はマネージャーのエンジンの乱数の種で決まる）。
   *
   * @param manager パラメーターのマネージャー
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
   * 順伝播する。autocast は開かずに呼ぶ（どこで丸めるかはクラスの説明のとおり）。段の区切りごとに、順伝播の種類が持つ計測点
   * （{@link FitProbe}）に切れ目を記録する。区切りの途中の配列は、区切りを出るときに閉じる（{@link Context#scoped}）。
   *
   * @param store パラメーターの値の取り出し元
   * @param inputs 入力（作った配列は入力のマネージャーに付く。記録する Attention の配列もその下に付くので、逆伝播を終えるまで
   *     閉じないこと）
   * @param pass 順伝播の種類
   * @return 出力
   * @throws IllegalArgumentException 入力の形式がネットワークの形式と違うとき
   * @throws IllegalStateException 記録する Attention が、既に順伝播を 1 回記録しているとき
   */
  public Outputs forward(ParameterStore store, NetworkInputs inputs, Pass pass) {
    return forward(store, inputs, pass, Context.Intermediates.CLOSE);
  }

  /**
   * 区切りの途中の配列の扱いを選んで順伝播する（{@link #forward(ParameterStore, NetworkInputs, Pass)}）。
   *
   * @param store パラメーターの値の取り出し元
   * @param inputs 入力
   * @param pass 順伝播の種類
   * @param intermediates 区切りの途中の配列の扱い
   * @return 出力
   */
  Outputs forward(
      ParameterStore store, NetworkInputs inputs, Pass pass, Context.Intermediates intermediates) {
    if (inputs.format() != format) {
      throw new IllegalArgumentException(
          "the inputs are " + inputs.format().variant() + " rows, the network reads " + format.variant());
    }
    FitProbe probe = pass.probe();
    AttentionTape tape = pass.tape();
    if (tape != null) {
      tape.open(inputs.manager());
    }
    Context context =
        new Context(
            store,
            inputs.manager(),
            pass.precision(),
            pass.training(),
            tape,
            probe,
            intermediates);
    BatchIndex index = inputs.index(pass);
    probe.mark(Stage.BATCH_INDEX);
    Backbone backbone = backbone(context, inputs, index);
    NDArray states = backbone.states();
    NDArray decision = states.get(":, {}, :", TokenType.ROUND.first(format));
    NDArray transitions = candidate.forward(context, inputs, index, states);
    if (reread != null) {
      transitions = reread.forward(context, inputs, index, transitions, states, backbone.events());
    }
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
    return new Outputs(
        scores, index.legal.array(), index.rowActions, index.discardValueSlots, auxiliary);
  }

  /**
   * バックボーンの出力。
   *
   * @param states 最後の正規化の後の局・席・牌 [行, 局・席・牌, H]
   * @param events 全てのトークンを読む最後の層が出力したイベント [行, 最大のイベント数, H]。記録があるときは、最初の局・席・牌の層に
   *     入る葉から切り出したもの。後で読まない形か、イベントの無いバッチでは null
   */
  private record Backbone(NDArray states, NDArray events) {}

  /**
   * バックボーン: 埋め込み → L 層 → 最後の正規化。局・席・牌の層は、局・席・牌だけを更新して返す。イベントは全てのトークンを読む最後の層の
   * 出力のまま残し、後の層のキーとバリューと、候補層の 2 段目が読む。記録があれば、全てのトークンを読む最後の層の出力を、最初の局・席・牌の
   * 層の入力のところで 1 回だけ葉にし、イベントはその葉から切り出す。こうすると、イベントを読む全ての所の勾配がその葉に溜まってから、
   * 前の層へ 1 回だけ流れる。後の局・席・牌の層の入力は、前の層の出力とイベントを並べて葉にする。
   */
  private Backbone backbone(Context context, NetworkInputs inputs, BatchIndex index) {
    FitProbe probe = context.probe();
    NDArray tokens = context.scoped(scope -> tokens(scope, inputs, index));
    probe.mark(Stage.EMBED);
    NDArray codes = relationCodes(context, inputs, index);
    probe.mark(Stage.RELATION);
    NDArray mask = index.tokenMask.array();
    int layer = 0;
    for (; roles[layer] == LayerRole.ALL_TOKENS; layer++) {
      tokens = encoders[layer].forward(context, tokens, mask, codes, index.tokens);
    }
    NDArray entry = context.cut(tokens, Stage.O_FFN_BACKWARD, layer - 1);
    NDArray events =
        keepEvents && index.maxEvents > 0
            ? entry.get(new NDIndex(":, {}:", format.fixedTokens()))
            : null;
    EncoderLayer.FixedTokens fixed = encoders[layer].fixed(context, entry, mask, codes);
    for (layer++; layer < encoders.length; layer++) {
      entry = context.cut(fixed.withEvents(events), Stage.O_FFN_BACKWARD, layer - 1);
      fixed = encoders[layer].fixed(context, entry, mask, codes);
    }
    // Attention の記録があれば、バックボーンの出力は記録が作る正規化の葉（型ごとのゲインなら 2 つの葉を並べたもの）になり、
    // 損失の逆伝播はヘッドの側（Stage.HEADS_BACKWARD）で止まる
    NDArray states = finalNorm.apply(context, fixed, encoders.length - 1);
    probe.mark(Stage.FINAL_NORM);
    return new Backbone(states, reread == null ? null : events);
  }

  /** 埋め込んだトークン [行, S, H]（局・席・牌の後に、行のイベントをパディング付きで並べる）。 */
  private NDArray tokens(Context context, NetworkInputs inputs, BatchIndex index) {
    int rows = inputs.rows();
    long hidden = shape.hidden();
    NDList parts = new NDList();
    for (TokenType type : TokenType.values()) {
      if (type == TokenType.EVENT) {
        if (index.maxEvents > 0) {
          NDArray events = embedding.embed(context, inputs, index, type);
          parts.add(
              NDArrays.scatterRows(events, index.eventSlots.array(), (long) rows * index.maxEvents)
                  .reshape(rows, index.maxEvents, hidden));
        }
      } else {
        parts.add(
            embedding.embed(context, inputs, index, type).reshape(rows, type.perRow(format), hidden));
      }
    }
    return NDArrays.concat(parts, 1);
  }

  /**
   * 関係のコード [行, S, S, 2]（INT16）。GPU の学習で Attention の記録の演算がコードも作るなら、自前の演算
   * （{@link AttentionTape#relationCodes}）で作る。ほか（CPU・djl-rocm の Attention・推論）は eager で作る。どちらでも同じビットになる。
   */
  private static NDArray relationCodes(Context context, NetworkInputs inputs, BatchIndex index) {
    RelationCodes kernel = context.tape() == null ? null : context.tape().relationCodes();
    return kernel == null ? RelationClass.codes(inputs, index) : RelationClass.codes(kernel, inputs, index);
  }

  /** 補助のヘッド（候補の結果と読み）。 */
  private Outputs.Auxiliary auxiliary(
      Context context,
      BatchIndex index,
      NDArray representations,
      NDArray values,
      NDArray decision,
      NDArray states) {
    NDArray candidates = null;
    if (index.candidateCount > 0) {
      NDArray chosen = ValueGraphHead.aggregate(representations, values, index.candidates);
      candidates =
          outcome.forward(
              context,
              chosen,
              context.probeCut(
                  NDArrays.gatherRows(decision, index.candidateRows.array()), Stage.GATHER_BACKWARD, 0));
    }
    NDArray readings =
        index.readingCount == 0
            ? null
            : reading.forward(
                context,
                context.probeCut(
                    NDArrays.gatherRows(states, index.readingRows.array()), Stage.GATHER_BACKWARD, 0));
    return new Outputs.Auxiliary(candidates, readings);
  }

  /**
   * 推論で使う重み（HIP の WeightLayout の元）を返す。登録の順に並べ、補助のヘッドは除く。
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
   * このネットワークの重みに、同じ形のネットワークの重みを混ぜる（方策の記憶に使う）。混ぜるネットワークの重みを w′ として、
   * w ← (1 − η)·w + η·w′ とする（{@link ParameterMix#mix}）。
   *
   * @param other 混ぜるネットワーク
   * @param eta 混ぜる割合 η（(0, 1]。1 なら混ぜるネットワークの重みをそのまま写す）
   * @throws IllegalArgumentException η が範囲の外のとき、またはパラメーターの名前か形が違うとき
   */
  public void mixIn(PrismNetwork other, double eta) {
    ParameterMix.mix(getParameters(), other.getParameters(), eta);
  }

  /**
   * 重み（weights.bin）と model.json を、新しいディレクトリに書く。model.json には、ネットワークの形式、入力のフィンガープリント、
   * ヘッドのフィンガープリント、形を書く。
   *
   * @param directory 書く先（まだ無いこと）
   * @throws IOException 書けないとき、または書く先が既にあるとき
   */
  public void save(Path directory) throws IOException {
    CheckpointFiles.write(
        directory,
        this,
        new Metadata(ARCHITECTURE, format.fingerprint(), TargetSchema.of(format.variant()).fingerprint(), shape));
  }

  /**
   * 保存したネットワークの形を読む。ネットワークの形式とフィンガープリント（入力、ヘッド）を、{@link #readFormat} と同じく照合する。
   *
   * @param directory 保存したディレクトリ
   * @return 形
   * @throws IOException 読めないとき、または形式かフィンガープリントが合わないとき
   */
  public static PrismShape readShape(Path directory) throws IOException {
    return header(directory).shape();
  }

  /**
   * 保存したネットワークが読む入力の形式を返す。入力のフィンガープリントは、classpath にあるルールのモジュールの形式のどれかと合わなければ
   * ならない。ヘッドのフィンガープリントは、そのルールのものと合わなければならない。
   *
   * @param directory 保存したディレクトリ
   * @return 入力の形式
   * @throws IOException 読めないとき、またはフィンガープリントがどのルールの形式とも合わないとき
   */
  public static InputFormat readFormat(Path directory) throws IOException {
    return header(directory).format();
  }

  private static Header header(Path directory) throws IOException {
    Metadata metadata = CheckpointFiles.readHeader(directory, Metadata.class);
    if (metadata.shape() == null || !ARCHITECTURE.equals(metadata.architecture())) {
      throw new IOException("Not a PRISM model: " + directory);
    }
    InputFormat format = null;
    for (PrismVariant module : PrismVariant.all()) {
      if (module.input().fingerprint().equals(metadata.inputSchema())) {
        format = module.input();
      }
    }
    if (format == null) {
      throw new IOException("PRISM input schema mismatch " + directory);
    }
    if (!TargetSchema.of(format.variant()).fingerprint().equals(metadata.headSchema())) {
      throw new IOException("PRISM head schema mismatch " + directory);
    }
    PrismShape shape =
        new PrismShape(
            metadata.shape().hidden(),
            metadata.shape().ffn(),
            metadata.shape().gate(),
            metadata.shape().depth(),
            metadata.shape().typed());
    return new Header(format, shape);
  }

  /**
   * 保存したネットワークを読む。
   *
   * @param directory 保存したディレクトリ
   * @param manager パラメーターのマネージャー
   * @return ネットワーク
   * @throws IOException 読めないとき、形式・フィンガープリント・重みが合わないとき、または読んだ重みの形が組んだネットワークと違うとき
   */
  public static PrismNetwork load(Path directory, NDManager manager) throws IOException {
    Header header = header(directory);
    PrismNetwork network = new PrismNetwork(header.format(), header.shape());
    network.initialize(manager);
    Map<String, Shape> built = new HashMap<>();
    for (Pair<String, Parameter> pair : network.getParameters()) {
      built.put(pair.getKey(), pair.getValue().getArray().getShape());
    }
    CheckpointFiles.loadParameters(directory, network, manager);
    // DJL は名前だけを照合し、形は読んだ配列で上書きするので、組んだときの形と全て照合する
    for (Pair<String, Parameter> pair : network.getParameters()) {
      Shape loaded = pair.getValue().getArray().getShape();
      if (!loaded.equals(built.get(pair.getKey()))) {
        throw new IOException(
            "PRISM weight "
                + pair.getKey()
                + " has shape "
                + loaded
                + ", the network needs "
                + built.get(pair.getKey())
                + " in "
                + directory);
      }
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
