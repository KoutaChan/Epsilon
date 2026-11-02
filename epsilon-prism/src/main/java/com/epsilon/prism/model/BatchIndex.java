package com.epsilon.prism.model;

import ai.djl.ndarray.types.Shape;
import com.epsilon.ai.model.MahjongTileRelation;
import com.epsilon.prism.Limits;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.model.ValueGraphHead.Composition;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 1 回の順伝播の索引。行と手の索引（ホストの写し）を確かめ、トークンの並べ替え・候補の集約・価値グラフの合成に要る添字をホストで作り、
 * {@link Staging} に載せる。デバイスの配列は区画（{@link Staging.Section}）で持ち、送った後に {@link Staging.Section#array} で読む。
 * 送り方は 2 つ: 順伝播がその場で作って入力の置き場 {@link NetworkInputs#uploads} から同期で写す（{@link #send}）か、学習の歩の作り手が
 * 前もって作って網の入力・損失の入力と一緒に非同期で写す（{@link StagedBatch}）。デバイスの上では値で形が決まる演算（nonzero など）を
 * 使わない。
 *
 * <p>手の値の作り方は {@link Composition} が役割から決める。代表でない双子は代表の遷移を読む。
 */
final class BatchIndex {

  private static final ActionRole[] ROLES = ActionRole.values();
  private static final TokenType[] TOKEN_TYPES = TokenType.values();

  final int rows;

  /** 行ごとの手の数。 */
  final int[] rowActions;

  final int maxEvents;
  final int tokens;
  final int actionCapacity;

  /** 出来事のトークンを [行 × 最大の出来事] へ散らす位置（INT64、出来事が 0 個なら null）。 */
  final Staging.Section eventSlots;

  /** 関係の属性を [行 × S] へ散らす位置（INT64）。 */
  final Staging.Section tokenSlots;

  /** トークンの在否 [行, S]（FP32）。 */
  final Staging.Section tokenMask;

  /** 位置ごとのトークンの型（{@link TokenType} の ordinal、INT32 [S]）。 */
  final Staging.Section tokenTypes;

  /** 牌の関係の類（{@link MahjongTileRelation} の ordinal、INT32 [34·34]）。 */
  final Staging.Section tileRelations;

  /** 遷移の行（INT32 [遷移]）。 */
  final Staging.Section transitionRows;

  /** 手を [行 × 手の容量] へ散らす位置（INT64）。 */
  final Staging.Section actionSlots;

  /** 合法手の印 [行, 手の容量]（BOOLEAN）。 */
  final Staging.Section legal;

  /**
   * 値を B が出す打牌の手の印（ホスト、[行 × 手の容量]）。行の定数は B の中にあるので、行の平均の錨（{@link PrismLoss#rowMean}）はこの手
   * だけで取る。
   */
  final boolean[] discardValueSlots;

  final int discardCount;
  final Staging.Section discardActions;
  final Staging.Section discardTransitions;

  final int gateCount;
  final Staging.Section gateActions;
  final Staging.Section gateKinds;
  final Staging.Section gateRows;

  /** 門の手 e_a の集約（門がなければ null）。 */
  final Aggregation gateChosen;

  /** 門の基準 e_base の集約。基準の手の z か、ē を使う門では行の ē（門がなければ null）。 */
  final Aggregation gateBase;

  final int riichiCount;
  final Staging.Section riichiActions;
  final Staging.Section riichiBaseTransitions;
  final Staging.Section riichiGates;

  /** 立直のダマ x が C にあるか（学習だけ、推論では null）。 */
  final Staging.Section riichiAnchors;

  final int responseCount;
  final Staging.Section responseActions;
  final Staging.Section responseGates;

  final int turnCount;
  final Staging.Section turnActions;
  final Staging.Section turnGates;
  final Staging.Section turnRows;

  /** M を取る手（打牌と立直）の印 [行, 手の容量]（BOOLEAN）。 */
  final Staging.Section pivotMask;

  /** M を取る手がある行で 1 [行, 1]（FP32）。 */
  final Staging.Section pivotPresent;

  /** C の印 [行, 手の容量]（学習だけ、推論では null）。 */
  final Staging.Section measured;

  final int candidateCount;

  /** 候補の手 e_a の集約（候補がなければ null）。 */
  final Aggregation candidates;

  final Staging.Section candidateRows;

  final int readingCount;
  final Staging.Section readingRows;

  /** 欄の並べ方ごとの列の添字の区画 {埋め込む列, 名前空間の先頭, 印の列}。 */
  private final Map<FieldPack, Staging.Section[]> columns = new IdentityHashMap<>();

  /** 作った C の印（推論では null。{@link #builtFor} が順伝播の種類と照らす）。 */
  private final boolean[] measuredFlags;

  /** 作った補助の頭の教師（なければ null）。 */
  private final AuxiliaryLabels labels;

  /**
   * 索引をその場で作り、入力の置き場から同期で送る。
   *
   * @param inputs 入力（置き場 {@link NetworkInputs#uploads} を持つ）
   * @param pass 順伝播の種類
   * @return 送った索引
   * @throws IllegalArgumentException 索引が並びの約束を外れるとき
   */
  static BatchIndex send(NetworkInputs inputs, Pass pass) {
    Staging staging = new Staging();
    BatchIndex index = new BatchIndex(inputs, pass.measured(), pass.auxiliary(), staging);
    staging.send(inputs.uploads(), inputs.manager());
    return index;
  }

  /**
   * 索引をホストで作って載せる（送るのは staging の持ち主）。
   *
   * @param inputs 塊の数と索引のホストの写し
   * @param measuredFlags 学習の C の印（塊の手の順。推論では null）
   * @param labels 補助の頭の教師（補助の頭を計算しないなら null）
   * @param staging 区画を載せる組
   * @throws IllegalArgumentException 索引が並びの約束を外れるとき
   */
  BatchIndex(
      HostColumns inputs, boolean[] measuredFlags, AuxiliaryLabels labels, Staging staging) {
    Structure structure = new Structure(inputs);
    rows = inputs.rows();
    rowActions = structure.actionCount;
    this.measuredFlags = measuredFlags;
    this.labels = labels;

    // トークン
    int events = inputs.events();
    int fixed = TokenType.FIXED_TOKENS;
    int widest = 0;
    for (int r = 0; r < rows; r++) {
      widest = Math.max(widest, structure.eventCount[r]);
    }
    maxEvents = widest;
    tokens = fixed + maxEvents;
    long[] eventSlot = new long[events];
    float[] mask = new float[rows * tokens];
    for (int r = 0; r < rows; r++) {
      for (int j = 0; j < structure.eventCount[r]; j++) {
        eventSlot[structure.firstEvent[r] + j] = (long) r * maxEvents + j;
      }
      for (int j = 0; j < fixed + structure.eventCount[r]; j++) {
        mask[r * tokens + j] = 1f;
      }
    }
    // 関係の属性（Axis.TOKEN）は書き手（BatchPlan・InputWriter）の並び: 型ごとの区画を TokenType の順に続ける
    // [局 R | 席 4R | 牌 34R | 出来事 N_E]。出来事の区画は行の firstEvent の順。網の行の中の位置は TokenType.first() + 番号。
    long[] tokenSlot = new long[rows * fixed + events];
    int token = 0;
    for (TokenType type : TOKEN_TYPES) {
      if (type == TokenType.EVENT) {
        for (int r = 0; r < rows; r++) {
          for (int j = 0; j < structure.eventCount[r]; j++) {
            tokenSlot[rows * fixed + structure.firstEvent[r] + j] =
                (long) r * tokens + type.first() + j;
          }
        }
      } else {
        for (int r = 0; r < rows; r++) {
          for (int k = 0; k < type.perRow(); k++) {
            tokenSlot[token++] = (long) r * tokens + type.first() + k;
          }
        }
      }
    }
    int[] types = new int[tokens];
    for (int j = 0; j < tokens; j++) {
      types[j] = TokenType.at(j).ordinal();
    }
    long[] relationIds = MahjongTileRelation.relationIds();
    int[] relations = new int[relationIds.length];
    for (int i = 0; i < relations.length; i++) {
      relations[i] = Math.toIntExact(relationIds[i]);
    }
    eventSlots = events == 0 ? null : staging.add(eventSlot, new Shape(events));
    tokenSlots = staging.add(tokenSlot, new Shape(tokenSlot.length));
    tokenMask = staging.add(mask, new Shape(rows, tokens));
    tokenTypes = staging.add(types, new Shape(tokens));
    tileRelations = staging.add(relations, new Shape(relations.length));

    // 遷移
    int[] transitionRow = new int[inputs.transitions()];
    for (int r = 0; r < rows; r++) {
      for (int t = 0; t < structure.transitionCount[r]; t++) {
        transitionRow[structure.firstTransitionOfRow[r] + t] = r;
      }
    }
    transitionRows = staging.add(transitionRow, new Shape(transitionRow.length));

    // 手の並べ方と価値グラフ
    int actions = inputs.actions();
    int capacity = 0;
    for (int r = 0; r < rows; r++) {
      capacity = Math.max(capacity, structure.actionCount[r]);
    }
    actionCapacity = capacity;
    if (measuredFlags != null && measuredFlags.length != actions) {
      throw new IllegalArgumentException(
          "measured has " + measuredFlags.length + " entries for " + actions + " actions");
    }
    long[] actionSlot = new long[actions];
    boolean[] legalSlot = new boolean[rows * capacity];
    discardValueSlots = new boolean[rows * capacity];
    boolean[] pivot = new boolean[rows * capacity];
    float[] pivotRow = new float[rows];
    boolean[] measuredSlot = measuredFlags == null ? null : new boolean[rows * capacity];
    Lists lists = new Lists();
    for (int a = 0; a < actions; a++) {
      int r = structure.row[a];
      int slot = r * capacity + a - structure.firstAction[r];
      actionSlot[a] = slot;
      legalSlot[slot] = true;
      if (measuredSlot != null) {
        measuredSlot[slot] = measuredFlags[a];
      }
      int representative = structure.representative[a];
      Composition composition = Composition.of(structure.role[a]);
      switch (composition) {
        case DISCARD_VALUE -> {
          discardValueSlots[slot] = true;
          lists.discardActions.add(a);
          lists.discardTransitions.add(structure.firstTransition(representative));
          pivot[slot] = true;
          pivotRow[r] = 1f;
        }
        case RIICHI_GATE -> {
          int base = structure.base(a, ActionRole.DISCARD);
          lists.riichiActions.add(a);
          lists.riichiBaseTransitions.add(structure.firstTransition(structure.representative[base]));
          lists.riichiGates.add(lists.gateActions.size());
          lists.riichiAnchors.add(measuredFlags != null && measuredFlags[base]);
          lists.addGate(a, structure, base, false);
          pivot[slot] = true;
          pivotRow[r] = 1f;
        }
        case RESPONSE_GATE -> {
          int base = structure.base(a, ActionRole.PASS);
          lists.responseActions.add(a);
          lists.responseGates.add(lists.gateActions.size());
          lists.addGate(a, structure, base, false);
        }
        case TURN_GATE -> {
          lists.turnActions.add(a);
          lists.turnGates.add(lists.gateActions.size());
          lists.turnRows.add(r);
          lists.addGate(a, structure, -1, true);
        }
        case ZERO -> {}
      }
    }
    actionSlots = staging.add(actionSlot, new Shape(actions));
    legal = staging.add(legalSlot, new Shape(rows, capacity));
    pivotMask = staging.add(pivot, new Shape(rows, capacity));
    pivotPresent = staging.add(pivotRow, new Shape(rows, 1));
    measured =
        measuredSlot == null ? null : staging.add(measuredSlot, new Shape(rows, capacity));

    discardCount = lists.discardActions.size();
    discardActions = staging.addLongs(lists.discardActions);
    discardTransitions = staging.addLongs(lists.discardTransitions);

    gateCount = lists.gateActions.size();
    gateActions = staging.addLongs(lists.gateActions);
    gateKinds = staging.addLongs(lists.gateKinds);
    gateRows = staging.addLongs(lists.gateRows);

    // ē: 行ごとの代表の打牌の遷移（行の中の手の順）
    turnCount = lists.turnActions.size();
    int[] discardStart = new int[rows + 1];
    for (int a = 0; a < actions; a++) {
      if (structure.role[a] == ActionRole.DISCARD && structure.representative[a] == a) {
        discardStart[structure.row[a] + 1]++;
      }
    }
    int bar = 0;
    for (int r = 0; r < rows; r++) {
      bar = Math.max(bar, discardStart[r + 1]);
      discardStart[r + 1] += discardStart[r];
    }
    int[] discardTransition = new int[discardStart[rows]];
    int[] filled = Arrays.copyOf(discardStart, rows);
    for (int a = 0; a < actions; a++) {
      if (structure.role[a] == ActionRole.DISCARD && structure.representative[a] == a) {
        discardTransition[filled[structure.row[a]]++] = structure.firstTransition(a);
      }
    }
    for (int index = 0; index < turnCount; index++) {
      int r = lists.turnRows.get(index);
      if (discardStart[r + 1] == discardStart[r]) {
        throw new IllegalArgumentException("turn gate in a row without discards");
      }
    }
    if (gateCount > 0) {
      Aggregation.Builder chosen = new Aggregation.Builder(gateCount);
      Aggregation.Builder base = new Aggregation.Builder(gateCount);
      for (int g = 0; g < gateCount; g++) {
        int action = lists.gateActions.get(g);
        chosen.add(structure.firstTransition(action), structure.transitionsOf(action));
        int baseTransition = lists.gateBaseTransitions.get(g);
        if (baseTransition == Lists.BAR) {
          int r = lists.gateRows.get(g);
          base.add(discardTransition, discardStart[r], discardStart[r + 1] - discardStart[r]);
        } else {
          base.add(baseTransition, 1);
        }
      }
      gateChosen = chosen.build(1, staging);
      // ē の幅は塊の全ての行の代表の打牌の数の最大。GPU の和の順は幅で決まるので、ē は行の打牌と
      // 塊だけで決まり、どの行に ē の門があるかに依らない（詰めるのは ē の門の行だけで、費用は小さい）
      gateBase = base.build(bar, staging);
    } else {
      gateChosen = null;
      gateBase = null;
    }

    riichiCount = lists.riichiActions.size();
    riichiActions = staging.addLongs(lists.riichiActions);
    riichiBaseTransitions = staging.addLongs(lists.riichiBaseTransitions);
    riichiGates = staging.addLongs(lists.riichiGates);
    riichiAnchors =
        riichiCount == 0 || measuredFlags == null
            ? null
            : staging.add(toBooleans(lists.riichiAnchors), new Shape(riichiCount));
    responseCount = lists.responseActions.size();
    responseActions = staging.addLongs(lists.responseActions);
    responseGates = staging.addLongs(lists.responseGates);
    turnActions = staging.addLongs(lists.turnActions);
    turnGates = staging.addLongs(lists.turnGates);
    turnRows = staging.addLongs(lists.turnRows);

    // 補助の頭（学習で教師があるときだけ）
    int[] candidates = labels == null ? new int[0] : labels.candidateActions();
    candidateCount = candidates.length;
    for (int action : candidates) {
      Objects.checkIndex(action, actions);
      if (!TargetSchema.hasCandidateTargets(structure.role[action])) {
        throw new IllegalArgumentException(
            "action " + action + " (" + structure.role[action] + ") has no candidate targets");
      }
    }
    if (candidateCount > 0) {
      Aggregation.Builder chosen = new Aggregation.Builder(candidateCount);
      long[] candidateRow = new long[candidateCount];
      for (int c = 0; c < candidateCount; c++) {
        int action = candidates[c];
        chosen.add(structure.firstTransition(action), structure.transitionsOf(action));
        candidateRow[c] = structure.row[action];
      }
      this.candidates = chosen.build(1, staging);
      candidateRows = staging.add(candidateRow, new Shape(candidateCount));
    } else {
      this.candidates = null;
      candidateRows = null;
    }
    int[] reading = labels == null ? new int[0] : labels.readingRows();
    readingCount = reading.length;
    long[] readingRow = new long[readingCount];
    for (int i = 0; i < readingCount; i++) {
      readingRow[i] = Objects.checkIndex(reading[i], rows);
    }
    readingRows =
        readingCount == 0 ? null : staging.add(readingRow, new Shape(readingCount));

    // 欄の並べ方の列の添字（定数だが、転送を 1 本にまとめるためここで載せる）
    for (FieldPack pack : FieldPack.ALL) {
      columns.put(
          pack,
          new Staging.Section[] {
            staging.add(pack.embeddedColumns(), new Shape(pack.embeddedColumns().length)),
            staging.add(pack.namespaceOffsets(), new Shape(1, pack.namespaceOffsets().length)),
            staging.add(pack.flagColumns(), new Shape(pack.flagColumns().length))
          });
    }
  }

  /**
   * この索引が順伝播の種類と同じ C の印・補助の頭の教師で作ったものかを返す。
   *
   * @param pass 順伝播の種類
   * @return 同じ配列から作ったなら true
   */
  boolean builtFor(Pass pass) {
    return pass.measured() == measuredFlags && pass.auxiliary() == labels;
  }

  /**
   * 欄の並べ方の列の添字を返す。
   *
   * @param pack 並べ方
   * @return デバイスの上の列の添字
   */
  FieldPack.Columns columns(FieldPack pack) {
    Staging.Section[] sections = Objects.requireNonNull(columns.get(pack), "unknown field pack");
    return new FieldPack.Columns(sections[0].array(), sections[1].array(), sections[2].array());
  }

  private static boolean[] toBooleans(List<Boolean> values) {
    boolean[] result = new boolean[values.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  /**
   * 手ごとの遷移の集約 e_m = Σ_t sg(softmax_t(B(t)/τ))·z_t の添字（{@link ValueGraphHead#aggregate}）。
   * 手は 2 つの組に分ける。遷移が 1 つの手は重みが厳密に 1 なので z_t を直に読み、2 つ以上の手だけを
   * 幅をそろえて詰める。詰め物はその手の最初の遷移・印 0 で、重みは厳密に 0。詰め物の宛先を手ごとに
   * 散らすのは、逆の原子的な加算を 1 つの遷移の行に集めないため。
   */
  static final class Aggregation {

    /** 手の数。 */
    final int count;

    /** 遷移が 1 つの手（直の組）の数。 */
    final int directCount;

    /** 遷移が 2 つ以上の手（詰めた組）の数。 */
    final int spreadCount;

    /** 詰めた組の幅（組の最大の遷移の数と、作り手が決めた下限の大きい方）。 */
    final int width;

    /** 直の組の遷移（INT64 [直の組]、組が空なら null）。 */
    final Staging.Section directTransitions;

    /** 詰めた組の遷移（INT64 [詰めた組 × 幅]、組が空なら null）。 */
    final Staging.Section spreadTransitions;

    /** 詰めた組の遷移の在否（FP32 [詰めた組, 幅]、組が空なら null）。 */
    final Staging.Section spreadMask;

    /**
     * 手ごとの、直の組の後に詰めた組を続けた列の中の位置（INT64 [手]）。両方の組があるときだけ持つ。
     * 片方の組だけなら、その組の順が手の順なので null。
     */
    final Staging.Section order;

    private Aggregation(int[] transitions, int[] ends, int minimumWidth, Staging staging) {
      count = ends.length;
      int direct = 0;
      int widest = minimumWidth;
      for (int m = 0, start = 0; m < count; start = ends[m++]) {
        int length = ends[m] - start;
        if (length == 1) {
          direct++;
        } else {
          widest = Math.max(widest, length);
        }
      }
      directCount = direct;
      spreadCount = count - direct;
      width = widest;
      long[] directTransition = new long[directCount];
      long[] spreadTransition = new long[spreadCount * width];
      float[] spreadSlot = new float[spreadCount * width];
      long[] position = new long[count];
      int d = 0;
      int k = 0;
      for (int m = 0, start = 0; m < count; start = ends[m++]) {
        int length = ends[m] - start;
        if (length == 1) {
          position[m] = d;
          directTransition[d++] = transitions[start];
        } else {
          position[m] = directCount + k;
          int from = k++ * width;
          Arrays.fill(spreadTransition, from, from + width, transitions[start]);
          for (int t = 0; t < length; t++) {
            spreadTransition[from + t] = transitions[start + t];
          }
          Arrays.fill(spreadSlot, from, from + length, 1f);
        }
      }
      directTransitions =
          directCount == 0 ? null : staging.add(directTransition, new Shape(directCount));
      spreadTransitions =
          spreadCount == 0
              ? null
              : staging.add(spreadTransition, new Shape(spreadTransition.length));
      spreadMask = spreadCount == 0 ? null : staging.add(spreadSlot, new Shape(spreadCount, width));
      order = directCount == 0 || spreadCount == 0 ? null : staging.add(position, new Shape(count));
    }

    /** 手を順に集める。 */
    static final class Builder {

      private final int[] ends;
      private int[] transitions;
      private int members;
      private int size;

      /**
       * @param count 手の数
       */
      Builder(int count) {
        ends = new int[count];
        transitions = new int[count];
      }

      /** 連続した遷移 [first, first + length) の手を足す。 */
      void add(int first, int length) {
        reserve(length);
        for (int t = 0; t < length; t++) {
          transitions[size + t] = first + t;
        }
        close(length);
      }

      /** 遷移の列 source[from, from + length) の手を足す。 */
      void add(int[] source, int from, int length) {
        reserve(length);
        System.arraycopy(source, from, transitions, size, length);
        close(length);
      }

      /**
       * 添字を作って載せる。
       *
       * @param minimumWidth 詰めた組の幅の下限
       * @param staging 区画を載せる組
       * @return 集約の添字
       * @throws IllegalStateException 足した手の数が作るときの数と違うとき
       */
      Aggregation build(int minimumWidth, Staging staging) {
        if (members != ends.length) {
          throw new IllegalStateException(
              "aggregation has " + members + " of " + ends.length + " actions");
        }
        return new Aggregation(transitions, ends, minimumWidth, staging);
      }

      private void reserve(int length) {
        if (length < 1) {
          throw new IllegalArgumentException("aggregated action without transitions");
        }
        if (size + length > transitions.length) {
          transitions = Arrays.copyOf(transitions, Math.max(size + length, transitions.length * 2));
        }
      }

      private void close(int length) {
        size += length;
        ends[members++] = size;
      }
    }
  }

  /** 行と手の索引のホストの写しを読み、並びの約束を確かめる。 */
  private static final class Structure {

    final int[] firstEvent;
    final int[] eventCount;
    final int[] firstAction;
    final int[] actionCount;
    final int[] firstTransitionOfRow;
    final int[] transitionCount;
    final int[] row;
    final int[] actionFirstTransition;
    final int[] actionTransitions;
    final ActionRole[] role;
    final int[] baseAction;
    final int[] representative;

    Structure(HostColumns inputs) {
      int rows = inputs.rows();
      firstEvent = column(inputs, rows, RowColumn.FIRST_EVENT);
      eventCount = column(inputs, rows, RowColumn.EVENTS);
      firstAction = column(inputs, rows, RowColumn.FIRST_ACTION);
      actionCount = column(inputs, rows, RowColumn.ACTIONS);
      firstTransitionOfRow = column(inputs, rows, RowColumn.FIRST_TRANSITION);
      transitionCount = column(inputs, rows, RowColumn.TRANSITIONS);
      contiguous("events", firstEvent, eventCount, inputs.events(), 0, InputSchema.EVENT_CAPACITY);
      contiguous("actions", firstAction, actionCount, inputs.actions(), 1, Limits.MAX_LEGAL_ACTIONS);
      contiguous(
          "transitions",
          firstTransitionOfRow,
          transitionCount,
          inputs.transitions(),
          1,
          Limits.MAX_TRANSITIONS);
      int actions = inputs.actions();
      row = new int[actions];
      actionFirstTransition = new int[actions];
      actionTransitions = new int[actions];
      role = new ActionRole[actions];
      baseAction = new int[actions];
      representative = new int[actions];
      for (int r = 0; r < rows; r++) {
        for (int a = firstAction[r]; a < firstAction[r] + actionCount[r]; a++) {
          if (inputs.action(a, ActionColumn.ROW) != r) {
            throw new IllegalArgumentException("action " + a + " does not belong to row " + r);
          }
          row[a] = r;
          actionFirstTransition[a] = inputs.action(a, ActionColumn.FIRST_TRANSITION);
          actionTransitions[a] = inputs.action(a, ActionColumn.TRANSITIONS);
          role[a] = ROLES[Objects.checkIndex(inputs.action(a, ActionColumn.ROLE), ROLES.length)];
          baseAction[a] = inputs.action(a, ActionColumn.BASE);
          representative[a] = inputs.action(a, ActionColumn.REPRESENTATIVE);
          if (!inRow(representative[a], r)) {
            throw new IllegalArgumentException("action " + a + " has a representative outside its row");
          }
          if (baseAction[a] != -1 && !inRow(baseAction[a], r)) {
            throw new IllegalArgumentException("action " + a + " has a base outside its row");
          }
        }
      }
      for (int a = 0; a < actions; a++) {
        int rep = representative[a];
        if (representative[rep] != rep || role[rep] != role[a]) {
          throw new IllegalArgumentException("action " + a + " has an invalid representative");
        }
        int r = row[a];
        if (rep == a) {
          int first = actionFirstTransition[a];
          int end = first + actionTransitions[a];
          if (actionTransitions[a] < 1
              || first < firstTransitionOfRow[r]
              || end > firstTransitionOfRow[r] + transitionCount[r]) {
            throw new IllegalArgumentException("action " + a + " has transitions outside its row");
          }
          if (role[a] == ActionRole.DISCARD && actionTransitions[a] != 1) {
            throw new IllegalArgumentException("discard " + a + " must have one transition");
          }
        } else if (actionTransitions[a] != 0) {
          throw new IllegalArgumentException("twin " + a + " must point to its representative");
        }
      }
    }

    /** 手の代表の最初の遷移。 */
    int firstTransition(int action) {
      return actionFirstTransition[representative[action]];
    }

    /** 手の代表の遷移の数。 */
    int transitionsOf(int action) {
      return actionTransitions[representative[action]];
    }

    /**
     * 基準の手を確かめて返す。
     *
     * @param action 手
     * @param expected 基準の手の役割
     * @return 基準の手
     */
    int base(int action, ActionRole expected) {
      int base = baseAction[action];
      if (base < 0 || role[base] != expected) {
        throw new IllegalArgumentException(
            role[action] + " action " + action + " needs a " + expected + " base");
      }
      return base;
    }

    private boolean inRow(int action, int r) {
      return action >= firstAction[r] && action < firstAction[r] + actionCount[r];
    }

    private static int[] column(HostColumns inputs, int rows, RowColumn column) {
      int[] values = new int[rows];
      for (int r = 0; r < rows; r++) {
        values[r] = inputs.row(r, column);
      }
      return values;
    }

    private static void contiguous(
        String what, int[] first, int[] count, int total, int minimum, int maximum) {
      int next = 0;
      for (int r = 0; r < first.length; r++) {
        if (first[r] != next || count[r] < minimum || count[r] > maximum) {
          throw new IllegalArgumentException(what + " of row " + r + " are not contiguous");
        }
        next += count[r];
      }
      if (next != total) {
        throw new IllegalArgumentException(what + " sum to " + next + ", expected " + total);
      }
    }
  }

  /** 役割ごとの手の組を集める。 */
  private static final class Lists {

    /** ē を基準にする門の印（遷移の番号にならない値）。 */
    static final int BAR = -1;

    final List<Integer> discardActions = new ArrayList<>();
    final List<Integer> discardTransitions = new ArrayList<>();
    final List<Integer> gateActions = new ArrayList<>();
    final List<Integer> gateKinds = new ArrayList<>();
    final List<Integer> gateRows = new ArrayList<>();

    /** 門の基準の手の最初の遷移（ē を使う門は {@link #BAR}）。 */
    final List<Integer> gateBaseTransitions = new ArrayList<>();

    final List<Integer> riichiActions = new ArrayList<>();
    final List<Integer> riichiBaseTransitions = new ArrayList<>();
    final List<Integer> riichiGates = new ArrayList<>();
    final List<Boolean> riichiAnchors = new ArrayList<>();
    final List<Integer> responseActions = new ArrayList<>();
    final List<Integer> responseGates = new ArrayList<>();
    final List<Integer> turnActions = new ArrayList<>();
    final List<Integer> turnGates = new ArrayList<>();
    final List<Integer> turnRows = new ArrayList<>();

    void addGate(int action, Structure structure, int base, boolean usesBar) {
      gateActions.add(action);
      gateKinds.add(ValueGraphHead.gateIndex(structure.role[action]));
      gateRows.add(structure.row[action]);
      gateBaseTransitions.add(usesBar ? BAR : structure.firstTransition(base));
    }
  }

}
