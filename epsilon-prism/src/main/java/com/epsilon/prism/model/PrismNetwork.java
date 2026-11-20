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
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.kernel.CandidateStages;
import com.epsilon.prism.kernel.EncoderLayerShape;
import com.epsilon.prism.kernel.LayerRole;
import com.epsilon.prism.kernel.ModelSchema;
import com.epsilon.prism.kernel.RelationCodes;
import com.epsilon.prism.variant.PrismVariant;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * PRISM のネットワーク。局面の表現を遷移ごとの Cross-Attention で読み、価値グラフのヘッドで各手のスコアを求める。
 * 局の最終状態を局面全体の表現に使う。
 *
 * <p>子ブロックの登録順は重みの保存形式の一部である。補助のヘッドは学習だけで使い、
 * {@link #inferenceParameters()} には含めない。
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
  private static final Gson JSON = new Gson();

  /** model.json の中身。 */
  private record Metadata(
      String architecture, String inputSchema, String headSchema, JsonObject shape) {}

  /**
   * model.json の形。型ごとの FFN を持たない形では typed を書かない。
   *
   * @param hidden 隠れ状態の幅 H
   * @param ffn FFN の幅 F
   * @param gate ヘッドの隠れ層の幅 G
   * @param depth バックボーンの層の数 L
   * @param typed 型ごとの FFN の幅（持たない形では null）
   */
  private record SavedShape(int hidden, int ffn, int gate, int depth, SavedTyped typed) {

    static JsonObject json(PrismShape shape) {
      return JSON.toJsonTree(
              switch (shape) {
                case PrismShape.V4 v4 -> v4;
                case PrismShape.V2 v2 ->
                    new SavedShape(v2.hidden(), v2.ffn(), v2.gate(), v2.depth(), null);
                case PrismShape.V3 v3 ->
                    new SavedShape(
                        v3.hidden(),
                        v3.ffn(),
                        v3.gate(),
                        v3.depth(),
                        new SavedTyped(v3.tile(), v3.context(), v3.candidate(), v3.reread()));
              })
          .getAsJsonObject();
    }

    /**
     * 形式の名前と合わせて形を作る。{@link PrismArchitecture#V2} の名前で型ごとの FFN を持つ形は、{@link PrismShape.V3} として読む。
     *
     * @param architecture model.json の architecture
     * @return 形
     * @throws IllegalArgumentException 名前がどの形式でもないとき、{@link PrismArchitecture#V3} の名前で型ごとの FFN が無いとき、
     *     型ごとの FFN を持つのに深さが {@link PrismShape.V3#DEPTH} でないとき、または幅が形の決まりを外れるとき
     */
    PrismShape shape(String architecture) {
      PrismArchitecture named = PrismArchitecture.byId(architecture);
      if (typed == null) {
        if (named == PrismArchitecture.V3) {
          throw new IllegalArgumentException(
              named.id() + " needs typed FFNs, but the shape has none");
        }
        return new PrismShape.V2(hidden, ffn, gate, depth);
      }
      if (depth != PrismShape.V3.DEPTH) {
        throw new IllegalArgumentException(
            "typed FFNs need depth " + PrismShape.V3.DEPTH + ", but the shape has depth " + depth);
      }
      return new PrismShape.V3(
          hidden, ffn, gate, typed.tile(), typed.context(), typed.candidate(), typed.reread());
    }
  }

  /**
   * model.json の型ごとの FFN の幅。
   *
   * @param tile 牌のトークンの FFN の幅
   * @param context 局・席のトークンの FFN の幅
   * @param candidate 候補層の FFN の幅
   * @param reread 候補層の 2 段目を持つなら true
   */
  private record SavedTyped(int tile, int context, int candidate, boolean reread) {}

  /** 保存したネットワークの入力の形式と形。 */
  public record Header(InputFormat format, PrismShape shape) {}

  private final ModelSchema.Model declaration;
  private final InputFormat format;
  private final PrismShape shape;

  /** 埋め込みの直後で全トークンを更新する層の数。 */
  private final int allTokenLayers;

  /** 後の局・席・牌の層のキーとバリュー、または候補層の 2 段目がイベントを読むなら true。 */
  private final boolean keepEvents;

  @FunctionalInterface
  private interface SecondStage {
    NDArray read(
        Context context, NetworkInputs inputs, BatchIndex index, NDArray start, Backbone backbone);
  }

  private final TokenEmbedding embedding;
  private final EncoderLayer[] encoders;
  private final RmsNorm finalNorm;
  private final CandidateLayer candidate;

  private final SecondStage secondStage;
  private final DiscardValue discardValue;
  private final ValueGraphHead gates;
  private final ReadingHeads reading;
  private final OutcomeHeads outcome;

  /**
   * ネットワークを組む（パラメーターは {@link #initialize(NDManager)} で作る）。
   *
   * @param baseFormat 対局ルールの入力形式。この形が必要とする形式をここから選ぶ。
   * @param shape 形
   */
  public PrismNetwork(InputFormat baseFormat, PrismShape shape) {
    InputFormat format = shape.inputFormat(baseFormat);
    this.format = format;
    this.shape = shape;
    embedding = addChildBlock("stateEmbedding", new TokenEmbedding(shape, format));
    List<EncoderLayerShape> layers = shape.layers();
    encoders = new EncoderLayer[layers.size()];
    for (int layer = 0; layer < encoders.length; layer++) {
      encoders[layer] = addChildBlock("encoder" + layer, shape.encoder(layer, format));
    }
    allTokenLayers =
        (int) layers.stream().filter(layer -> layer.role() == LayerRole.ALL_TOKENS).count();
    finalNorm = addChildBlock("finalNorm", new RmsNorm((PrismShape.TokenTrunk) shape));
    CandidateLayer.Features features = shape.candidateFeatures(format);
    candidate = addChildBlock("candidate", new CandidateLayer(shape, format, features));
    CandidateReread reread =
        shape.reread() ? addChildBlock("reread", new CandidateReread(shape, format)) : null;
    secondStage =
        switch (shape.candidateStages()) {
          case ONE -> (context, inputs, index, start, backbone) -> start;
          case REREAD ->
              (context, inputs, index, start, backbone) ->
                  reread.fromEncoded(
                      context, inputs, index, start, backbone.states(), backbone.events());
        };
    keepEvents =
        shape.candidateStages() == CandidateStages.REREAD || encoders.length - allTokenLayers > 1;
    discardValue = addChildBlock("discardValue", new DiscardValue(shape, format));
    gates = addChildBlock("gates", new ValueGraphHead(shape, format));
    reading = addChildBlock("reading", new ReadingHeads(shape, format));
    outcome = addChildBlock("outcome", new OutcomeHeads(shape, TargetSchema.of(format.variant())));
    // 共通の重みを同じ乱数列で初期化してから、追加の重みを作る。
    if (features instanceof Module extra) addChildBlock("candidateFeatures", extra);
    declaration = ModelDeclaration.of(shape, format, inferenceParameters());
  }

  /** 入力・層・候補と元の推論パラメーターを保存時の順に返す。 */
  public ModelSchema.Model declaration() {
    return declaration;
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
   * 順伝播する。autocast は開かずに呼ぶ（どこで丸めるかはクラスの説明のとおり）。区切りの途中の配列は、区切りを出るときに閉じる
   * （{@link Context#scoped}）。
   *
   * @param store パラメーターの値の取り出し元
   * @param inputs 入力（作った配列は入力のマネージャーに付く。記録する Attention の配列もその下に付くので、逆伝播を終えるまで
   *     閉じないこと）
   * @param pass 順伝播の種類
   * @return 出力
   * @throws IllegalArgumentException 入力形式が違うとき、または記録に対応しない形に Attention の記録を渡したとき
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
          "the inputs are "
              + inputs.format().variant()
              + " rows ("
              + inputs.format().fingerprint()
              + "), the network reads "
              + format.variant()
              + " ("
              + format.fingerprint()
              + ")");
    }
    AttentionTape tape = pass.tape();
    if (tape != null) {
      if (!shape.supportsAttentionTape()) {
        throw new IllegalArgumentException(
            "This shape does not support an attention tape: " + shape);
      }
      tape.open(inputs.manager());
    }
    Context context =
        new Context(
            store, inputs.manager(), pass.precision(), pass.training(), tape, intermediates);
    BatchIndex index = inputs.index(pass);
    NDArray tokens = context.scoped(scope -> tokens(scope, inputs, index));
    NDArray codes = relationCodes(context, inputs, index, index.tokens);
    NDArray input = context.cut(tokens);
    NDArray entry = input;
    NDArray mask = index.tokenMask.array();
    for (int layer = 0; layer < allTokenLayers; layer++) {
      tokens = encoders[layer].forward(context, entry, mask, codes, index.tokens);
      entry = context.cut(tokens);
    }
    Backbone backbone = fixedLayers(context, index, entry, codes);
    NDArray states = backbone.states();
    if (tape != null && !tape.backbone()) {
      states = context.cutCandidateInput(states);
    }
    NDArray decision = states.get(new NDIndex(":, {}, :", TokenType.ROUND.first(format)));
    NDArray transitions = candidate.forward(context, inputs, index, states);
    transitions = secondStage.read(context, inputs, index, transitions, backbone);
    NDArray values = discardValue.forward(context, transitions, inputs.get(Tensor.DISCARD_EXACT));
    NDArray representations = context.toFloat(transitions);
    NDArray scores =
        gates.scores(
            context, index, representations, values, decision, inputs.get(Tensor.ACTION_EXACT));
    Outputs.Auxiliary auxiliary =
        pass.auxiliary() == null
            ? null
            : auxiliary(context, index, representations, values, decision, states);
    return new Outputs(
        scores, index.legal.array(), index.rowActions, index.discardValueSlots, auxiliary);
  }

  /**
   * バックボーンの出力。
   *
   * @param states 最後の正規化の後の局・席・牌 [行, 局・席・牌, H]
   * @param events 後続の固定トークン層や V3 の候補が読むイベント。イベントがない場合は null
   */
  record Backbone(NDArray states, NDArray events) {}

  /**
   * 固定トークンだけを更新し、後続の層ではイベントをキーとバリューに使う。イベントが層 0 のクエリにならない形では、
   * 専用の残差 MLP を一度通す。記録中は同じ入力の葉を共有し、そこから派生した全経路の勾配をまとめる。
   */
  Backbone fixedLayers(Context context, BatchIndex index, NDArray entry, NDArray codes) {
    NDArray mask = index.tokenMask.array();
    int layer = allTokenLayers;
    NDArray events = keepEvents ? events(entry, index) : null;
    EncoderLayer.FixedTokens fixed = encoders[layer].fixed(context, entry, mask, codes);
    for (layer++; layer < encoders.length; layer++) {
      entry = context.cut(fixed.withEvents(events));
      fixed = encoders[layer].fixed(context, entry, mask, codes);
    }
    // Attention の記録があれば、バックボーンの出力は記録が作る正規化の葉（型ごとのゲインなら 2 つの葉を並べたもの）になり、
    // 損失の逆伝播はヘッドの側で止まる
    NDArray states = finalNorm.apply(context, fixed);
    return new Backbone(states, events);
  }

  private NDArray events(NDArray tokens, BatchIndex index) {
    return index.maxEvents == 0 ? null : tokens.get(new NDIndex(":, {}:", format.fixedTokens()));
  }

  /** 埋め込んだトークン [行, S, H]（局・席・牌の後に、行のイベントをパディング付きで並べる）。 */
  NDArray tokens(Context context, NetworkInputs inputs, BatchIndex index) {
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
            embedding
                .embed(context, inputs, index, type)
                .reshape(rows, type.perRow(format), hidden));
      }
    }
    return NDArrays.concat(parts, 1);
  }

  /**
   * 関係のコード [行, クエリ数, S, 2]（INT16）。GPU の学習で Attention の記録の演算がコードも作るなら、自前の演算
   * （{@link AttentionTape#relationCodes}）で作る。それ以外は eager で作る。どちらでも同じビットになる。
   */
  private static NDArray relationCodes(
      Context context, NetworkInputs inputs, BatchIndex index, int queries) {
    RelationCodes kernel = context.tape() == null ? null : context.tape().relationCodes();
    return kernel == null
        ? RelationClass.codes(inputs, index, queries)
        : RelationClass.codes(kernel, inputs, index);
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
              context, chosen, NDArrays.gatherRows(decision, index.candidateRows.array()));
    }
    NDArray readings = null;
    if (index.readingCount > 0) {
      NDArray rows = index.readingRows.array();
      NDArray selected = NDArrays.gatherRows(states, rows);
      readings = reading.forward(context, selected);
    }
    return new Outputs.Auxiliary(candidates, readings);
  }

  /**
   * 推論で使う重み（重みの配置表の作成に使う）を返す。登録の順に並べ、補助のヘッドは除く。
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
   * @throws IllegalArgumentException η が範囲の外のとき、またはネットワークの形やパラメーターの名前・形が違うとき
   */
  public void mixIn(PrismNetwork other, double eta) {
    if (!shape.equals(other.shape)) {
      throw new IllegalArgumentException(
          "Cannot mix networks with different shapes: " + shape + " and " + other.shape);
    }
    ParameterMix.mix(getParameters(), other.getParameters(), eta);
  }

  /**
   * 重み（weights.bin）と model.json を、新しいディレクトリに書く。model.json には、ネットワークの形式（{@link PrismArchitecture}）、
   * 入力のフィンガープリント、ヘッドのフィンガープリント、形を書く。
   *
   * @param directory 書く先（まだ無いこと）
   * @throws IOException 書けないとき、または書く先が既にあるとき
   */
  public void save(Path directory) throws IOException {
    CheckpointFiles.write(
        directory,
        this,
        new Metadata(
            shape.architecture().id(),
            format.fingerprint(),
            TargetSchema.of(format.variant()).fingerprint(),
            SavedShape.json(shape)));
  }

  /**
   * 保存したネットワークの形を読む。ネットワークの形式とフィンガープリント（入力、ヘッド）を、{@link #readFormat} と同じく照合する。
   *
   * @param directory 保存したディレクトリ
   * @return 形
   * @throws IOException 読めないとき、または形式かフィンガープリントが合わないとき
   */
  public static PrismShape readShape(Path directory) throws IOException {
    return readHeader(directory).shape();
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
    return readHeader(directory).format();
  }

  /** 入力形式と形を同じモデルの記述から読む。 */
  public static Header readHeader(Path directory) throws IOException {
    Metadata metadata = CheckpointFiles.readHeader(directory, Metadata.class);
    if (metadata.shape() == null || metadata.architecture() == null) {
      throw new IOException("Not a PRISM model: " + directory);
    }
    PrismShape shape;
    try {
      PrismArchitecture architecture = PrismArchitecture.byId(metadata.architecture());
      JsonObject record = metadata.shape();
      shape =
          switch (architecture) {
            case V2, V3 -> JSON.fromJson(record, SavedShape.class).shape(metadata.architecture());
            case V4 -> architecture.readShape(JSON, record);
          };
    } catch (RuntimeException e) {
      throw new IOException("Not a readable PRISM model: " + directory + ": " + e.getMessage(), e);
    }
    InputFormat format = null;
    for (PrismVariant module : PrismVariant.all()) {
      InputFormat candidate = shape.inputFormat(module.input());
      if (candidate.fingerprint().equals(metadata.inputSchema())) {
        format = candidate;
      }
    }
    if (format == null) {
      throw new IOException("PRISM input schema mismatch " + directory);
    }
    if (!TargetSchema.of(format.variant()).fingerprint().equals(metadata.headSchema())) {
      throw new IOException("PRISM head schema mismatch " + directory);
    }
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
    Header header = readHeader(directory);
    PrismNetwork network = new PrismNetwork(header.format(), header.shape());
    network.readWeights(directory, manager);
    return network;
  }

  /** 検証済みのヘッダーから組み立てたネットワークに、保存されたパラメーターを読み込む。 */
  public void readWeights(Path directory, NDManager manager) throws IOException {
    initialize(manager);
    Map<String, Shape> built = new HashMap<>();
    for (Pair<String, Parameter> pair : getParameters()) {
      built.put(pair.getKey(), pair.getValue().getArray().getShape());
    }
    CheckpointFiles.loadParameters(directory, this, manager);
    // DJL は名前だけを照合し、形は読んだ配列で上書きするので、組んだときの形と全て照合する
    for (Pair<String, Parameter> pair : getParameters()) {
      Shape loaded = pair.getValue().getArray().getShape();
      if (pair.getValue().getArray().getDataType() != DataType.FLOAT32)
        throw new IOException("Expected FP32 parameter: " + pair.getKey());
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
