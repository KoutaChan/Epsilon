package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipLayer;
import com.epsilon.prism.hip.LayerRole;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * eager のネットワークで段ごとの値を求める。HIP のカーネルを 1 つずつ比べるテストで、参照の値として使う。ネットワークの forward と同じ
 * 部品を同じ順で計算し、最後に forward の s とビットまで照合する（ずれれば例外）。値は HIP の作業領域と同じ並びの float で返す。トークンの
 * 軸は型ごとの範囲に分かれ、局 R・席 S·R・牌 T·R・イベント N_E の順に並ぶ（席と牌の数はバッチの入力の形式で決まる）。全ての形（層の種類、
 * 型ごとの FFN、候補層の 2 段目）と、四人打ち・三人打ちの形式を受け付ける。
 */
public final class EagerStages {

  private EagerStages() {}

  /**
   * 段ごとの値。F は 1 行あたりの局・席・牌のトークンの数（四人打ち 39、三人打ち 31）。
   *
   * @param embedded 埋め込み x0 [トークン, H]
   * @param statics 遷移の静的な部分 t_s [遷移, H]
   * @param attention 層ごとの Attention の出力（O 射影の前）。全トークンの層は [トークン, A]、局・席・牌の層は [F·R, A]
   * @param layers 層ごとの出力。全トークンの層は [トークン, H]、局・席・牌の層は [F·R, H]（最後の層は最後の正規化の前）
   * @param eventKeyValues 局・席・牌の層ごとの、イベントのキーとバリュー [イベント, 2A]（イベントごとに [k | v]。最後の全トークンの層の
   *     出力から作る）
   * @param states 最後の正規化の後 y [F·R, H]
   * @param memory Cross-Attention のキーとバリュー [R, F, 2A]
   * @param rereadMemory 候補層の 2 段目のキーとバリュー（[R, F, 2A] の後に [イベント, 2A]。2 段目のない形では null）
   * @param candidates 候補層の 1 段目の出力 z1 [遷移, H]（1 段の形では z と同じ）
   * @param transitions 遷移の表現 z [遷移, H]
   * @param values B [遷移]
   * @param scores s（バッチの手の順）
   */
  public record Stages(
      float[] embedded,
      float[] statics,
      List<float[]> attention,
      List<float[]> layers,
      List<float[]> eventKeyValues,
      float[] states,
      float[] memory,
      float[] rereadMemory,
      float[] candidates,
      float[] transitions,
      float[] values,
      float[] scores) {}

  /**
   * 確定したバッチからネットワークの入力を作る。各領域はそのまま写す（インデックスは、ライターと同じ型ごとの領域の並びを前提に読む）。
   *
   * @param manager 配列のマネージャー
   * @param batch 確定したバッチ
   * @return ネットワークの入力
   */
  public static NetworkInputs inputs(NDManager manager, HostBatch batch) {
    BatchPlan plan = batch.plan();
    InputFormat format = plan.format();
    int rows = plan.rows();
    int events = plan.count(Axis.EVENT);
    int transitions = plan.count(Axis.TRANSITION);
    int actions = plan.count(Axis.ACTION);
    Map<Tensor, ByteBuffer> sections = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      int count = NetworkInputs.count(format, tensor.axis(), rows, events, transitions, actions);
      int bytes = count * format.width(tensor) * tensor.element().bytes();
      ByteBuffer copy = ByteBuffer.allocateDirect(Math.max(bytes, 1)).order(ByteOrder.nativeOrder());
      ByteBuffer source = batch.blob().duplicate();
      source.position(plan.offset(tensor)).limit(plan.offset(tensor) + bytes);
      copy.put(source.slice());
      copy.position(0);
      sections.put(tensor, copy);
    }
    return NetworkInputs.of(manager, PinnedUploads.on(manager), format, rows, events, transitions, actions, sections);
  }

  /** ネットワークの部品。登録の順に子から拾う。 */
  private record Parts(
      TokenEmbedding embedding,
      List<EncoderLayer> encoders,
      RmsNorm finalNorm,
      CandidateLayer candidate,
      CandidateReread reread,
      DiscardValue discardValue,
      ValueGraphHead gates) {

    static Parts of(PrismNetwork network) {
      TokenEmbedding embedding = null;
      List<EncoderLayer> encoders = new ArrayList<>();
      RmsNorm finalNorm = null;
      CandidateLayer candidate = null;
      CandidateReread reread = null;
      DiscardValue discardValue = null;
      ValueGraphHead gates = null;
      for (Pair<String, Block> child : network.getChildren()) {
        switch (child.getValue()) {
          case TokenEmbedding value -> embedding = value;
          case EncoderLayer value -> encoders.add(value);
          case RmsNorm value -> finalNorm = value;
          case CandidateLayer value -> candidate = value;
          case CandidateReread value -> reread = value;
          case DiscardValue value -> discardValue = value;
          case ValueGraphHead value -> gates = value;
          default -> {}
        }
      }
      return new Parts(embedding, encoders, finalNorm, candidate, reread, discardValue, gates);
    }
  }

  /** バッチの入力の形式と、行ごとのイベントの並び。値を型ごとの範囲の並びに直すために使う。 */
  private record Rows(InputFormat format, int rows, int[] firstEvent, int[] events, int totalEvents) {

    static Rows of(NetworkInputs inputs) {
      int rows = inputs.rows();
      int[] firstEvent = new int[rows];
      int[] events = new int[rows];
      int total = 0;
      for (int r = 0; r < rows; r++) {
        firstEvent[r] = inputs.row(r, RowColumn.FIRST_EVENT);
        events[r] = inputs.row(r, RowColumn.EVENTS);
        total += events[r];
      }
      return new Rows(inputs.format(), rows, firstEvent, events, total);
    }
  }

  /**
   * 段ごとの値を計算する（推論の順伝播）。
   *
   * @param network ネットワーク
   * @param store パラメーターの値の取り元
   * @param inputs 入力（{@link #inputs}）
   * @param precision 計算の型
   * @return 段ごとの値
   */
  public static Stages run(PrismNetwork network, ParameterStore store, NetworkInputs inputs, Precision precision) {
    InputFormat format = inputs.format();
    Pass pass = Pass.inference(precision);
    Context context = new Context(store, inputs.manager(), precision, false);
    BatchIndex index = BatchIndex.send(inputs, pass);
    Parts parts = Parts.of(network);
    PrismShape shape = network.shape();
    Rows rows = Rows.of(inputs);
    int hidden = shape.hidden();
    int attentionWidth = shape.attention();
    int fixed = format.fixedTokens();

    // 埋め込み（型ごと）と、行ごとに並べたトークン
    NDList typed = new NDList();
    NDList pieces = new NDList();
    for (TokenType type : TokenType.values()) {
      if (type == TokenType.EVENT) {
        if (index.maxEvents > 0) {
          NDArray events = parts.embedding().embed(context, inputs, index, type);
          typed.add(events);
          pieces.add(
              NDArrays.scatterRows(events, index.eventSlots.array(), (long) rows.rows() * index.maxEvents)
                  .reshape(rows.rows(), index.maxEvents, hidden));
        }
      } else {
        NDArray values = parts.embedding().embed(context, inputs, index, type);
        typed.add(values);
        pieces.add(values.reshape(rows.rows(), type.perRow(format), hidden));
      }
    }
    float[] embedded = floats(NDArrays.concat(typed, 0));
    NDArray tokens = NDArrays.concat(pieces, 1);
    NDArray attributes =
        NDArrays.scatterRows(
                inputs.get(Tensor.RELATION).toType(DataType.INT32, false),
                index.tokenSlots.array(),
                (long) rows.rows() * index.tokens)
            .reshape(rows.rows(), index.tokens, format.relation().width());
    NDArray codes = RelationClass.codes(format, attributes, index.tokenTypes.array(), index.tileRelations.array());
    NDArray mask = index.tokenMask.array();

    // バックボーン: PrismNetwork と同じ層のメソッドで組む。全トークンの層は forward で、局・席・牌の層は fixed で局・席・牌を更新する。
    // イベントは最後の全トークンの層の出力のままにして、後の層でキーとバリューとして読む
    List<float[]> attention = new ArrayList<>();
    List<float[]> layers = new ArrayList<>();
    List<float[]> eventKeyValues = new ArrayList<>();
    List<HipLayer> roles = shape.layers();
    NDArray events = null;
    EncoderLayer.FixedTokens fixedTokens = null;
    for (int layer = 0; layer < parts.encoders().size(); layer++) {
      EncoderLayer encoder = parts.encoders().get(layer);
      Map<String, Parameter> parameters = parameters(encoder);
      if (roles.get(layer).role() == LayerRole.ALL_TOKENS) {
        NDArray attended = attend(context, shape, parameters, tokens, mask, codes, index.tokens);
        attention.add(sectioned(floats(attended), rows, index.tokens, attentionWidth, false));
        tokens = encoder.forward(context, tokens, mask, codes, index.tokens);
        layers.add(sectioned(floats(tokens), rows, index.tokens, hidden, false));
        continue;
      }
      if (fixedTokens == null) {
        events = index.maxEvents > 0 ? tokens.get(new NDIndex(":, {}:", fixed)) : null;
      }
      NDArray input = fixedTokens == null ? tokens : fixedTokens.withEvents(events);
      eventKeyValues.add(
          events == null
              ? new float[0]
              : eventSection(
                  floats(eventKeyValue(context, shape, parameters, events)),
                  rows,
                  index.maxEvents,
                  0,
                  2 * attentionWidth));
      NDArray attended = attend(context, shape, parameters, input, mask, codes, fixed);
      attention.add(sectioned(floats(attended), rows, fixed, attentionWidth, true));
      fixedTokens = encoder.fixed(context, input, mask, codes);
      layers.add(sectioned(floats(fixedTokens.withEvents(null)), rows, fixed, hidden, true));
    }
    NDArray states = parts.finalNorm().apply(context, fixedTokens, parts.encoders().size() - 1);
    float[] stateValues = sectioned(floats(states), rows, fixed, hidden, true);

    Map<String, Parameter> candidate = parameters(parts.candidate());
    NDArray packed =
        FieldPack.of(format).transition().pack(
            inputs.get(Tensor.TRANSITION_CATEGORY),
            index.columns(FieldPack.of(format).transition()),
            context.value(candidate.get("table")),
            context.toCompute(inputs.get(Tensor.TRANSITION_SCALAR)),
            context.toCompute(inputs.get(Tensor.DISCARD_EXACT)));
    float[] statics = floats(context.linear(packed, candidate.get("project"), candidate.get("projectBias")));
    float[] memory = floats(context.linear(states, candidate.get("keyValue"), candidate.get("keyValueBias")));
    NDArray start = parts.candidate().forward(context, inputs, index, states);
    float[] candidates = floats(start);
    NDArray transitions = start;
    float[] rereadMemory = null;
    if (parts.reread() != null) {
      rereadMemory = rereadMemory(context, shape, parameters(parts.reread()), states, events, rows, index.maxEvents);
      transitions = parts.reread().forward(context, inputs, index, start, states, events);
    }
    NDArray values = parts.discardValue().forward(context, transitions, inputs.get(Tensor.DISCARD_EXACT));
    NDArray decision = states.get(":, {}, :", TokenType.ROUND.first(format));
    NDArray scores =
        parts
            .gates()
            .scores(context, index, context.toFloat(transitions), values, decision, inputs.get(Tensor.ACTION_EXACT));
    float[] dense = scores.toFloatArray();
    float[] forward = network.forward(store, inputs, pass).scores().toFloatArray();
    for (int i = 0; i < dense.length; i++) {
      if (Float.floatToRawIntBits(dense[i]) != Float.floatToRawIntBits(forward[i])) {
        throw new IllegalStateException("eager stages diverged from PrismNetwork.forward at score " + i);
      }
    }
    float[] ordered = ordered(inputs, dense, index.actionCapacity);
    return new Stages(
        embedded,
        statics,
        attention,
        layers,
        eventKeyValues,
        stateValues,
        memory,
        rereadMemory,
        candidates,
        floats(transitions),
        values.toFloatArray(),
        ordered);
  }

  /**
   * forward の s だけを計算する（段ごとの値は作らない）。GPU の eager では、段の部品を順に呼ぶ場合と forward とでリダクションの順が違う
   * ことがあるので、s はこちらで取る。
   *
   * @param network ネットワーク
   * @param store パラメーターの値の取り元
   * @param inputs 入力（{@link #inputs}）
   * @param precision 計算の型
   * @return s（バッチの手の順）
   */
  public static float[] scores(PrismNetwork network, ParameterStore store, NetworkInputs inputs, Precision precision) {
    NDArray scores = network.forward(store, inputs, Pass.inference(precision)).scores();
    return ordered(inputs, scores.toFloatArray(), Math.toIntExact(scores.getShape().get(1)));
  }

  /** [行, 1 行あたりの手の上限] の形の s を、バッチの手の順に並べ直す。 */
  private static float[] ordered(NetworkInputs inputs, float[] dense, int capacity) {
    float[] ordered = new float[inputs.actions()];
    for (int r = 0; r < inputs.rows(); r++) {
      int first = inputs.row(r, RowColumn.FIRST_ACTION);
      int count = inputs.row(r, RowColumn.ACTIONS);
      for (int k = 0; k < count; k++) {
        ordered[first + k] = dense[r * capacity + k];
      }
    }
    return ordered;
  }

  /** ネットワークの集約の温度。HIP の形の記述にある τ と照合する。 */
  public static float tau() {
    return ValueGraphHead.TAU;
  }

  /** ネットワークの Attention の尺度。HIP の形の記述にある尺度と照合する。 */
  public static double scale() {
    return EncoderLayer.SCALE;
  }

  private static Map<String, Parameter> parameters(Block block) {
    Map<String, Parameter> parameters = new HashMap<>();
    for (Pair<String, Parameter> pair : block.getParameters()) {
      parameters.put(pair.getKey(), pair.getValue());
    }
    return parameters;
  }

  /**
   * EncoderLayer.forward の Attention の前半（O 射影の前の出力）を計算する。キーとバリューは入力の全てのトークンから、クエリは先頭の
   * {@code queries} 個のトークンから作る。
   */
  private static NDArray attend(
      Context context,
      PrismShape shape,
      Map<String, Parameter> parameters,
      NDArray tokens,
      NDArray mask,
      NDArray codes,
      int queries) {
    int attention = shape.attention();
    boolean all = queries == tokens.getShape().get(1);
    NDArray normalized = context.norm(tokens, parameters.get("attentionNorm"), shape.epsilon());
    NDArray weights = context.value(parameters.get("qkv"));
    NDArray biases = context.value(parameters.get("qkvBias"));
    NDArray keyValue =
        context.linear(
            normalized,
            context.slice(weights, attention, 3L * attention),
            context.slice(biases, attention, 3L * attention));
    NDArray query =
        context.linear(
            all ? normalized : normalized.get(new NDIndex(":, 0:{}", queries)),
            context.slice(weights, 0, attention),
            context.slice(biases, 0, attention));
    NDArray bias = context.fp32(parameters.get("relation"));
    NDArray table =
        context.own(
            bias.concat(
                context.working().zeros(new Shape(RelationClass.ROWS, attention), DataType.FLOAT32), 1));
    return NDArrays.packedRelationScaledDotProductAttention(
        query,
        keyValue,
        mask,
        all ? codes : codes.get(new NDIndex(":, 0:{}", queries)),
        table,
        shape.heads(),
        RelationClass.ENTRIES,
        EncoderLayer.SCALE);
  }

  /**
   * 局・席・牌の層で読むイベントの [k | v] を計算する。層の Attention の前の正規化をかけたイベントを、W_qkv の k・v の行で射影し、
   * [行, イベント, 2A] で返す。
   */
  private static NDArray eventKeyValue(
      Context context, PrismShape shape, Map<String, Parameter> parameters, NDArray events) {
    int attention = shape.attention();
    NDArray normalized = context.norm(events, parameters.get("attentionNorm"), shape.epsilon());
    return context.linear(
        normalized,
        context.slice(context.value(parameters.get("qkv")), attention, 3L * attention),
        context.slice(context.value(parameters.get("qkvBias")), attention, 3L * attention));
  }

  /**
   * 候補層の 2 段目のキーとバリューを計算する。局・席・牌は y を射影して [R, F, 2A] に、イベントは 2 段目のゲインで正規化してから射影して
   * [イベント, 2A] にし、この順に並べて返す。
   */
  private static float[] rereadMemory(
      Context context,
      PrismShape shape,
      Map<String, Parameter> parameters,
      NDArray states,
      NDArray events,
      Rows rows,
      int maxEvents) {
    int width = 2 * shape.attention();
    float[] fixed = floats(context.linear(states, parameters.get("keyValue"), parameters.get("keyValueBias")));
    float[] out = new float[fixed.length + rows.totalEvents() * width];
    System.arraycopy(fixed, 0, out, 0, fixed.length);
    if (events != null) {
      NDArray normalized = context.norm(events, parameters.get("eventNorm"), shape.epsilon());
      float[] eventValues =
          eventSection(
              floats(context.linear(normalized, parameters.get("keyValue"), parameters.get("keyValueBias"))),
              rows,
              maxEvents,
              0,
              width);
      System.arraycopy(eventValues, 0, out, fixed.length, eventValues.length);
    }
    return out;
  }

  /** [行, n, w] の行ごとの並びを、型ごとの範囲の並び（局・席・牌・イベント）に直す。{@code fixedOnly} なら局・席・牌だけを並べる。 */
  private static float[] sectioned(float[] values, Rows rows, int perRow, int width, boolean fixedOnly) {
    int seats = TokenType.SEAT.perRow(rows.format());
    int tiles = TokenType.TILE.perRow(rows.format());
    int fixed = rows.format().fixedTokens();
    int count = rows.rows();
    int total = fixedOnly ? fixed * count : fixed * count + rows.totalEvents();
    float[] out = new float[total * width];
    for (int r = 0; r < count; r++) {
      copy(values, out, (r * perRow) * width, r * width, width);
      for (int s = 0; s < seats; s++) {
        copy(values, out, (r * perRow + 1 + s) * width, (count + seats * r + s) * width, width);
      }
      for (int t = 0; t < tiles; t++) {
        copy(values, out, (r * perRow + 1 + seats + t) * width, ((1 + seats) * count + tiles * r + t) * width, width);
      }
    }
    if (!fixedOnly) {
      float[] events = eventSection(values, rows, perRow, fixed, width);
      System.arraycopy(events, 0, out, fixed * count * width, events.length);
    }
    return out;
  }

  /** [行, n, w] の行ごとのイベント（行の中の位置 {@code offset} から）を、イベントの範囲の並び [イベント, w] に直す。 */
  private static float[] eventSection(float[] values, Rows rows, int perRow, int offset, int width) {
    float[] out = new float[rows.totalEvents() * width];
    for (int r = 0; r < rows.rows(); r++) {
      for (int e = 0; e < rows.events()[r]; e++) {
        copy(values, out, (r * perRow + offset + e) * width, (rows.firstEvent()[r] + e) * width, width);
      }
    }
    return out;
  }

  private static void copy(float[] from, float[] to, int fromAt, int toAt, int width) {
    System.arraycopy(from, fromAt, to, toAt, width);
  }

  private static float[] floats(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }
}
