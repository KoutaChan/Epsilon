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
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * eager の網の段の境目の値（HIP のカーネルごとの一致の試験の参照）。網の forward と同じ部品・同じ順で計算し、最後に forward の s と
 * ビットで照らす（ずれれば例外）。値は HIP の作業域と同じ並び（トークンの軸は型ごとの区画: 局 R・席 4R・牌 34R・出来事 N_E）の float
 * で返す。
 */
public final class EagerStages {

  private EagerStages() {}

  /**
   * 段の境目の値。
   *
   * @param embedded 埋め込み x0 [トークン, H]
   * @param statics 遷移の静的な部分 t_s [遷移, H]
   * @param attention 層ごとの注意の出力（O 射影の前）。最後でない層は [トークン, A]、最後の層は [39R, A]
   * @param layers 層ごとの出力。最後でない層は [トークン, H]、最後の層は [39R, H]（最後の正規化の前）
   * @param states 最後の正規化の後 y [39R, H]
   * @param memory 交差注意の鍵と値 [R, 39, 2A]
   * @param transitions 遷移の表現 z [遷移, H]
   * @param values B [遷移]
   * @param scores s（塊の手の順）
   */
  public record Stages(
      float[] embedded,
      float[] statics,
      List<float[]> attention,
      List<float[]> layers,
      float[] states,
      float[] memory,
      float[] transitions,
      float[] values,
      float[] scores) {}

  /**
   * 封じた塊から網の入力を作る。区画はそのまま写す（網の索引は書き手と同じ型ごとの区画の並びを読む）。
   *
   * @param manager 配列の管理元
   * @param batch 封じた塊
   * @return 網の入力
   */
  public static NetworkInputs inputs(NDManager manager, HostBatch batch) {
    BatchPlan plan = batch.plan();
    int rows = plan.rows();
    int events = plan.count(Axis.EVENT);
    int transitions = plan.count(Axis.TRANSITION);
    int actions = plan.count(Axis.ACTION);
    Map<Tensor, ByteBuffer> sections = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      int count = NetworkInputs.count(tensor.axis(), rows, events, transitions, actions);
      int bytes = count * tensor.width() * tensor.element().bytes();
      ByteBuffer copy = ByteBuffer.allocateDirect(Math.max(bytes, 1)).order(ByteOrder.nativeOrder());
      ByteBuffer source = batch.blob().duplicate();
      source.position(plan.offset(tensor)).limit(plan.offset(tensor) + bytes);
      copy.put(source.slice());
      copy.position(0);
      sections.put(tensor, copy);
    }
    return NetworkInputs.of(
        manager, Uploads.on(manager), rows, events, transitions, actions, sections);
  }

  /**
   * 段の境目の値を計算する（推論の順伝播）。
   *
   * @param network 網
   * @param store パラメーターの値の取り元
   * @param inputs 入力（{@link #inputs}）
   * @param precision 計算の型
   * @return 段の値
   */
  public static Stages run(PrismNetwork network, ParameterStore store, NetworkInputs inputs, Precision precision) {
    Pass pass = Pass.inference(precision);
    Context context = new Context(store, inputs.manager(), precision, false);
    BatchIndex index = BatchIndex.send(inputs, pass);
    TokenEmbedding embedding = null;
    List<EncoderLayer> encoders = new ArrayList<>();
    RmsNorm finalNorm = null;
    CandidateLayer candidate = null;
    DiscardValue discardValue = null;
    ValueGraphHead gates = null;
    for (Pair<String, Block> child : network.getChildren()) {
      Block block = child.getValue();
      if (block instanceof TokenEmbedding value) {
        embedding = value;
      } else if (block instanceof EncoderLayer value) {
        encoders.add(value);
      } else if (block instanceof RmsNorm value) {
        finalNorm = value;
      } else if (block instanceof CandidateLayer value) {
        candidate = value;
      } else if (block instanceof DiscardValue value) {
        discardValue = value;
      } else if (block instanceof ValueGraphHead value) {
        gates = value;
      }
    }
    int rows = inputs.rows();
    long hidden = network.shape().hidden();
    int fixed = TokenType.FIXED_TOKENS;
    int[] firstEvent = new int[rows];
    int[] eventCounts = new int[rows];
    for (int r = 0; r < rows; r++) {
      firstEvent[r] = inputs.row(r, InputSchema.RowColumn.FIRST_EVENT);
      eventCounts[r] = inputs.row(r, InputSchema.RowColumn.EVENTS);
    }

    // 埋め込み（型ごと）と、行ごとに並べた流れ
    NDList typed = new NDList();
    NDList parts = new NDList();
    for (TokenType type : TokenType.values()) {
      if (type == TokenType.EVENT) {
        if (index.maxEvents > 0) {
          NDArray events = embedding.embed(context, inputs, index, type);
          typed.add(events);
          parts.add(
              NDArrays.scatterRows(events, index.eventSlots.array(), (long) rows * index.maxEvents)
                  .reshape(rows, index.maxEvents, hidden));
        }
      } else {
        NDArray values = embedding.embed(context, inputs, index, type);
        typed.add(values);
        parts.add(values.reshape(rows, type.perRow(), hidden));
      }
    }
    float[] embedded = floats(NDArrays.concat(typed, 0));
    NDArray tokens = NDArrays.concat(parts, 1);
    NDArray attributes =
        NDArrays.scatterRows(
                inputs.get(Tensor.RELATION).toType(DataType.INT32, false),
                index.tokenSlots.array(),
                (long) rows * index.tokens)
            .reshape(rows, index.tokens, InputSchema.RELATION.width());
    NDArray codes = RelationClass.codes(attributes, index.tokenTypes.array(), index.tileRelations.array());

    List<float[]> attention = new ArrayList<>();
    List<float[]> layers = new ArrayList<>();
    for (int layer = 0; layer < encoders.size(); layer++) {
      boolean last = layer == encoders.size() - 1;
      int queries = last ? fixed : index.tokens;
      EncoderLayer encoder = encoders.get(layer);
      NDArray attended = attend(context, network.shape(), encoder, tokens, index.tokenMask.array(), codes, queries);
      attention.add(sectioned(floats(attended), rows, queries, network.shape().attention(), firstEvent, eventCounts, last));
      tokens = encoder.forward(context, tokens, index.tokenMask.array(), codes, queries);
      layers.add(sectioned(floats(tokens), rows, queries, (int) hidden, firstEvent, eventCounts, last));
    }
    NDArray states = finalNorm.apply(context, tokens);
    float[] stateValues = sectioned(floats(states), rows, fixed, (int) hidden, firstEvent, eventCounts, true);

    Map<String, Parameter> parameters = new HashMap<>();
    for (Pair<String, Parameter> pair : candidate.getParameters()) {
      parameters.put(pair.getKey(), pair.getValue());
    }
    NDArray packed =
        FieldPack.TRANSITION.pack(
            inputs.get(Tensor.TRANSITION_CATEGORY),
            index.columns(FieldPack.TRANSITION),
            context.value(parameters.get("table")),
            context.toCompute(inputs.get(Tensor.TRANSITION_SCALAR)),
            context.toCompute(inputs.get(Tensor.DISCARD_EXACT)));
    float[] statics =
        floats(context.linear(packed, parameters.get("project"), parameters.get("projectBias")));
    float[] memory = floats(context.linear(states, parameters.get("keyValue"), parameters.get("keyValueBias")));
    NDArray transitions = candidate.forward(context, inputs, index, states);
    NDArray values =
        discardValue.forward(context, transitions, inputs.get(Tensor.DISCARD_EXACT));
    NDArray decision = states.get(":, {}, :", TokenType.ROUND.first());
    NDArray scores =
        gates.scores(
            context,
            index,
            context.toFloat(transitions),
            values,
            decision,
            inputs.get(Tensor.ACTION_EXACT));
    float[] dense = scores.toFloatArray();
    float[] forward = network.forward(store, inputs, pass).scores().toFloatArray();
    for (int i = 0; i < dense.length; i++) {
      if (Float.floatToRawIntBits(dense[i]) != Float.floatToRawIntBits(forward[i])) {
        throw new IllegalStateException("eager stages diverged from PrismNetwork.forward at score " + i);
      }
    }
    float[] ordered = new float[inputs.actions()];
    int capacity = index.actionCapacity;
    for (int r = 0; r < rows; r++) {
      int first = inputs.row(r, InputSchema.RowColumn.FIRST_ACTION);
      int count = inputs.row(r, InputSchema.RowColumn.ACTIONS);
      for (int k = 0; k < count; k++) {
        ordered[first + k] = dense[r * capacity + k];
      }
    }
    return new Stages(
        embedded,
        statics,
        attention,
        layers,
        stateValues,
        memory,
        floats(transitions),
        values.toFloatArray(),
        ordered);
  }

  /** 網の集約の温度（WeightLayout.TAU と照らす）。 */
  public static float tau() {
    return ValueGraphHead.TAU;
  }

  /** 網の注意の尺度（WeightLayout.SCALE と照らす）。 */
  public static double scale() {
    return EncoderLayer.SCALE;
  }

  /** EncoderLayer.forward の注意の半分（O 射影の前の出力）。 */
  private static NDArray attend(
      Context context,
      PrismShape shape,
      EncoderLayer layer,
      NDArray tokens,
      NDArray mask,
      NDArray codes,
      int queries) {
    Map<String, Parameter> parameters = new HashMap<>();
    for (Pair<String, Parameter> pair : layer.getParameters()) {
      parameters.put(pair.getKey(), pair.getValue());
    }
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

  /** [行, n, w] の行ごとの並びを、型ごとの区画の並び（局・席・牌・出来事）に直す。fixedOnly なら局・席・牌だけ。 */
  private static float[] sectioned(
      float[] values, int rows, int perRow, int width, int[] firstEvent, int[] events, boolean fixedOnly) {
    int seats = 4;
    int tiles = 34;
    int fixed = 1 + seats + tiles;
    int eventTotal = 0;
    for (int count : events) {
      eventTotal += count;
    }
    int total = fixedOnly ? fixed * rows : fixed * rows + eventTotal;
    float[] out = new float[total * width];
    for (int r = 0; r < rows; r++) {
      copy(values, out, (r * perRow) * width, r * width, width);
      for (int s = 0; s < seats; s++) {
        copy(values, out, (r * perRow + 1 + s) * width, (rows + seats * r + s) * width, width);
      }
      for (int t = 0; t < tiles; t++) {
        copy(values, out, (r * perRow + 1 + seats + t) * width, ((1 + seats) * rows + tiles * r + t) * width, width);
      }
      if (!fixedOnly) {
        for (int e = 0; e < events[r]; e++) {
          copy(values, out, (r * perRow + fixed + e) * width, (fixed * rows + firstEvent[r] + e) * width, width);
        }
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
