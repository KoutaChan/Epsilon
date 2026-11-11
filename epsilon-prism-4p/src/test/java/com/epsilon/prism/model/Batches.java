package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Element;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.TokenLayout;
import com.epsilon.prism4p.input.InputSchema;
import com.epsilon.prism4p.input.InputSchema.ActionRole;
import com.epsilon.prism4p.input.InputSchema.KeyBlock;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * テスト用のバッチとネットワークのヘルパー。フィールドは {@link TokenLayout} の語彙の中の乱数で作り、4 種の判断の行は手と遷移の形まで
 * 自分で組む（Tokenizer を通さない）。
 */
final class Batches {

  private static final TargetSchema HEADS = TargetSchema.of(Variant.TENHOU_4P);

  /** テスト用の小さな形（CPU で速く回す）。 */
  static PrismShape small(int depth) {
    return new PrismShape(64, 128, 64, depth, null);
  }

  /**
   * 1 つの手の形。
   *
   * @param type 手の種類
   * @param transitions 遷移の数（同値な手のうち代表でない手は 0）
   * @param base 基準の手（行の中の番号、なしは −1）
   * @param representative 同値な手の代表（行の中の番号）
   */
  record Move(Action.Type type, int transitions, int base, int representative) {}

  /**
   * 1 行の形。
   *
   * @param kind 判断の種類
   * @param events イベントの数
   * @param moves 手（合法手の順）
   */
  record Row(EngineDecisionKind kind, int events, List<Move> moves) {

    int transitions() {
      int total = 0;
      for (Move move : moves) {
        total += move.transitions();
      }
      return total;
    }
  }

  /** 手番: 打牌 x・y・z、立直 x（基準は x）、暗槓、ツモ、九種、y と同値な打牌。 */
  static Row turn(int events) {
    return new Row(
        EngineDecisionKind.TURN,
        events,
        List.of(
            new Move(Action.Type.DAHAI, 1, -1, 0),
            new Move(Action.Type.DAHAI, 1, -1, 1),
            new Move(Action.Type.DAHAI, 1, -1, 2),
            new Move(Action.Type.RIICHI_DAHAI, 1, 0, 3),
            new Move(Action.Type.ANKAN, 1, -1, 4),
            new Move(Action.Type.TSUMO_AGARI, 1, -1, 5),
            new Move(Action.Type.KYUSHU_KYUHAI, 1, -1, 6),
            new Move(Action.Type.DAHAI, 0, -1, 1)));
  }

  /** 応答: PASS、チー（鳴いた後の打牌 3）、ポン（2）、大明槓、ロン。基準は PASS。 */
  static Row response(int events) {
    return new Row(
        EngineDecisionKind.RESPONSE,
        events,
        List.of(
            new Move(Action.Type.PASS, 1, -1, 0),
            new Move(Action.Type.CHI, 3, 0, 1),
            new Move(Action.Type.PON, 2, 0, 2),
            new Move(Action.Type.DAIMINKAN, 1, 0, 3),
            new Move(Action.Type.RON_AGARI, 1, 0, 4)));
  }

  /** 槍槓: PASS とロン。 */
  static Row chankan(int events) {
    return new Row(
        EngineDecisionKind.CHANKAN,
        events,
        List.of(new Move(Action.Type.PASS, 1, -1, 0), new Move(Action.Type.RON_AGARI, 1, 0, 1)));
  }

  /** 鳴いた後の打牌: 打牌 4 つ。 */
  static Row postCall(int events) {
    List<Move> moves = new ArrayList<>();
    for (int m = 0; m < 4; m++) {
      moves.add(new Move(Action.Type.DAHAI, 1, -1, m));
    }
    return new Row(EngineDecisionKind.POST_CALL_DAHAI, events, moves);
  }

  /** 4 種の判断の行。 */
  static List<Row> allKinds(int events) {
    return List.of(turn(events), response(events + 1), chankan(events + 2), postCall(events + 3));
  }

  /**
   * バッチ（ホストの領域）。
   *
   * @param rows 行の形
   * @param events イベントの数
   * @param transitions 遷移の数
   * @param actions 手の数
   * @param sections 領域
   */
  record Batch(List<Row> rows, int events, int transitions, int actions, Map<Tensor, ByteBuffer> sections) {

    /** ネットワークの入力。 */
    NetworkInputs inputs(NDManager manager) {
      return NetworkInputs.of(
          manager, PinnedUploads.on(manager), InputSchema.FORMAT, rows.size(), events, transitions, actions, sections);
    }

    /** 行の最初の手の、バッチでの通し番号。 */
    int firstAction(int row) {
      int first = 0;
      for (int r = 0; r < row; r++) {
        first += rows.get(r).moves().size();
      }
      return first;
    }

    /** バッチでの通し番号で指した手の {@code ActionRole}。 */
    ActionRole role(int action) {
      for (Row row : rows) {
        if (action < row.moves().size()) {
          return roleOf(row.moves().get(action).type());
        }
        action -= row.moves().size();
      }
      throw new IndexOutOfBoundsException();
    }
  }

  /** 行の並びからバッチを作る。 */
  static Batch build(List<Row> rows, long seed) {
    Random random = new Random(seed);
    int events = 0;
    int transitions = 0;
    int actions = 0;
    for (Row row : rows) {
      events += row.events();
      transitions += row.transitions();
      actions += row.moves().size();
    }
    int count = rows.size();
    Map<Tensor, ByteBuffer> sections = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      int length = NetworkInputs.count(InputSchema.FORMAT, tensor.axis(), count, events, transitions, actions) * InputSchema.FORMAT.width(tensor);
      sections.put(
          tensor,
          ByteBuffer.allocateDirect(length * tensor.element().bytes()).order(ByteOrder.nativeOrder()));
    }
    categories(sections.get(Tensor.ROUND_CATEGORY), InputSchema.ROUND, random);
    categories(sections.get(Tensor.SEAT_CATEGORY), InputSchema.SEAT, random);
    categories(sections.get(Tensor.TILE_CATEGORY), InputSchema.TILE, random);
    categories(sections.get(Tensor.EVENT_CATEGORY), InputSchema.EVENT, random);
    categories(sections.get(Tensor.RELATION), InputSchema.RELATION, random);
    categories(sections.get(Tensor.TRANSITION_CATEGORY), InputSchema.TRANSITION, random);
    for (Tensor tensor : Tensor.values()) {
      ByteBuffer section = sections.get(tensor);
      if (tensor.element() == Element.BF16) {
        while (section.hasRemaining()) {
          section.putShort(bfloat16(random.nextFloat()));
        }
      } else if (tensor == Tensor.DISCARD_EXACT || tensor == Tensor.ACTION_EXACT) {
        while (section.hasRemaining()) {
          section.putFloat((float) random.nextGaussian() * 0.5f);
        }
      } else if (tensor == Tensor.TRANSITION_CODE) {
        while (section.hasRemaining()) {
          section.put((byte) random.nextInt(KeyBlock.CODE_COUNT));
        }
      }
    }
    ByteBuffer rowColumns = sections.get(Tensor.ROW_COLUMN);
    ByteBuffer actionColumns = sections.get(Tensor.ACTION_COLUMN);
    int firstEvent = 0;
    int firstAction = 0;
    int firstTransition = 0;
    for (int r = 0; r < count; r++) {
      Row row = rows.get(r);
      put(rowColumns, r, RowColumn.FIRST_EVENT, firstEvent);
      put(rowColumns, r, RowColumn.EVENTS, row.events());
      put(rowColumns, r, RowColumn.FIRST_ACTION, firstAction);
      put(rowColumns, r, RowColumn.ACTIONS, row.moves().size());
      put(rowColumns, r, RowColumn.FIRST_TRANSITION, firstTransition);
      put(rowColumns, r, RowColumn.TRANSITIONS, row.transitions());
      put(rowColumns, r, RowColumn.DECISION_KIND, row.kind().ordinal());
      int transition = firstTransition;
      for (int m = 0; m < row.moves().size(); m++) {
        Move move = row.moves().get(m);
        int action = firstAction + m;
        putAction(actionColumns, action, ActionColumn.ROW, r);
        putAction(actionColumns, action, ActionColumn.FIRST_TRANSITION, move.transitions() == 0 ? -1 : transition);
        putAction(actionColumns, action, ActionColumn.TRANSITIONS, move.transitions());
        putAction(actionColumns, action, ActionColumn.ROLE, roleOf(move.type()).ordinal());
        putAction(actionColumns, action, ActionColumn.BASE, move.base() < 0 ? -1 : firstAction + move.base());
        putAction(actionColumns, action, ActionColumn.REPRESENTATIVE, firstAction + move.representative());
        transition += move.transitions();
      }
      firstEvent += row.events();
      firstAction += row.moves().size();
      firstTransition += row.transitions();
    }
    for (ByteBuffer section : sections.values()) {
      section.rewind();
    }
    return new Batch(rows, events, transitions, actions, sections);
  }

  /** 全ての手を測った乱数の Q（λ = 0.1）。 */
  static List<ValueLabel> values(Batch batch, long seed) {
    Random random = new Random(seed);
    List<ValueLabel> labels = new ArrayList<>();
    for (Row row : batch.rows()) {
      float[] values = new float[row.moves().size()];
      for (int m = 0; m < values.length; m++) {
        values[m] = random.nextFloat() - 0.5f;
      }
      labels.add(new ValueLabel(values, 0.1f, 1f));
    }
    return labels;
  }

  /** 教師のある全ての候補と全ての行の補助の教師。 */
  static AuxiliaryLabels auxiliary(Batch batch, long seed) {
    Random random = new Random(seed);
    List<Integer> candidates = new ArrayList<>();
    for (int a = 0; a < batch.actions(); a++) {
      if (TargetSchema.hasCandidateTargets(batch.role(a))) {
        candidates.add(a);
      }
    }
    int n = candidates.size();
    int[] actions = new int[n];
    int[] outcomes = new int[n];
    float[] values = new float[n * HEADS.candidateWidth()];
    for (int c = 0; c < n; c++) {
      actions[c] = candidates.get(c);
      outcomes[c] = c % Outcome.values().length;
      int base = c * HEADS.candidateWidth();
      values[base + HEADS.offset(CandidateTarget.OUTCOME) + outcomes[c]] = 1f;
      values[base + HEADS.offset(CandidateTarget.TENPAI)] = random.nextInt(2);
      for (int y = 0; y < ScoringYaku.values().length; y++) {
        values[base + HEADS.offset(CandidateTarget.YAKU) + y] = random.nextFloat() < 0.1f ? 1f : 0f;
      }
      values[base + HEADS.offset(CandidateTarget.WIN_POINTS)] = 1000 + random.nextInt(11000);
      values[base + HEADS.offset(CandidateTarget.DEAL_IN_POINTS)] = 1000 + random.nextInt(11000);
      values[base + HEADS.offset(CandidateTarget.YAKU_TENPAI)] = random.nextInt(2);
      values[base + HEADS.offset(CandidateTarget.POINT_DELTA)] = random.nextInt(16001) - 8000;
      values[base + HEADS.offset(CandidateTarget.WINNER) + random.nextInt(3)] = 1f;
    }
    int rows = batch.rows().size();
    int[] readingRows = new int[rows];
    float[] reading = new float[rows * HEADS.opponents() * HEADS.readingWidth()];
    for (int r = 0; r < rows; r++) {
      readingRows[r] = r;
      for (int j = 0; j < HEADS.opponents(); j++) {
        int base = (r * HEADS.opponents() + j) * HEADS.readingWidth();
        reading[base + HEADS.offset(ReadingTarget.TENPAI)] = random.nextFloat() < 0.5f ? 1f : 0f;
        reading[base + HEADS.offset(ReadingTarget.SHANTEN) + random.nextInt(TargetSchema.SHANTEN_CLASSES)] = 1f;
        for (int t = 0; t < HEADS.width(ReadingTarget.DANGER); t++) {
          reading[base + HEADS.offset(ReadingTarget.DANGER) + t] = random.nextFloat() < 0.2f ? 1f : 0f;
          reading[base + HEADS.offset(ReadingTarget.WAITS) + t] = random.nextFloat() < 0.2f ? 1f : 0f;
          reading[base + HEADS.offset(ReadingTarget.DANGER_POINTS) + t] = 1000 + random.nextInt(11000);
        }
      }
    }
    return new AuxiliaryLabels(HEADS, actions, outcomes, values, readingRows, reading);
  }

  /** 全てのパラメーターに N(0, scale²) を足す（0 で始める層も動かし、s を 0 でなくする）。 */
  static void perturb(PrismNetwork network, float scale, long seed) {
    Random random = new Random(seed);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      float[] values = array.toFloatArray();
      for (int i = 0; i < values.length; i++) {
        values[i] += (float) random.nextGaussian() * scale;
      }
      array.set(values);
    }
  }

  /** パラメーターごとの勾配の写しを取り、勾配を 0 に戻す。 */
  static Map<String, float[]> takeGradients(PrismNetwork network) {
    Map<String, float[]> gradients = new LinkedHashMap<>();
    for (Pair<String, Parameter> pair : network.getParameters()) {
      try (NDArray gradient = pair.getValue().getArray().getGradient()) {
        gradients.put(pair.getKey(), gradient.toFloatArray());
        gradient.fillI(0);
      }
    }
    return gradients;
  }

  /** Adam の 1 ステップ（勾配は呼び出し側が作り、ここで 0 に戻す）。 */
  static void step(PrismNetwork network, Optimizer optimizer) {
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      try (NDArray gradient = array.getGradient()) {
        optimizer.update(pair.getValue().getId(), array, gradient);
        gradient.fillI(0);
      }
    }
  }

  /** 学習率を固定した Adam。 */
  static Optimizer adam(float rate) {
    return Optimizer.adam().optLearningRateTracker(Tracker.fixed(rate)).build();
  }

  static float maxAbs(float[] values) {
    float max = 0;
    for (float value : values) {
      max = Math.max(max, Math.abs(value));
    }
    return max;
  }

  private static void categories(ByteBuffer section, TokenLayout<?> layout, Random random) {
    int[] vocabularies = layout.columnVocabularies();
    int column = 0;
    while (section.hasRemaining()) {
      section.put((byte) random.nextInt(vocabularies[column]));
      column = (column + 1) % vocabularies.length;
    }
    section.rewind();
  }

  private static void put(ByteBuffer section, int row, RowColumn column, int value) {
    section.putInt((row * RowColumn.values().length + column.ordinal()) * Integer.BYTES, value);
  }

  private static void putAction(ByteBuffer section, int action, ActionColumn column, int value) {
    section.putInt((action * ActionColumn.values().length + column.ordinal()) * Integer.BYTES, value);
  }

  /** float を BF16 のビット列に丸める（最近接偶数）。 */
  private static short bfloat16(float value) {
    int bits = Float.floatToIntBits(value);
    return (short) ((bits + 0x7FFF + ((bits >>> 16) & 1)) >>> 16);
  }

  private Batches() {}

  /** 旧エンジンの手の種類に対応する {@code ActionRole}（同じ名前の新しい種類で引く）。 */
  private static ActionRole roleOf(Action.Type type) {
    return ActionRole.of(type);
  }
}
