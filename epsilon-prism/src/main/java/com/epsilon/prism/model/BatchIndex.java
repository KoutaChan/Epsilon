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
  final int gateWidth;
  final Staging.Section gateActions;
  final Staging.Section gateKinds;
  final Staging.Section gateRows;
  final Staging.Section gateTransitions;
  final Staging.Section gateTransitionMask;
  final Staging.Section gateBaseTransitions;
  final Staging.Section gateUsesBar;

  final int barWidth;
  final Staging.Section barTransitions;
  final Staging.Section barMask;

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
  final int candidateWidth;
  final Staging.Section candidateTransitions;
  final Staging.Section candidateMask;
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
    int width = 1;
    for (int g = 0; g < gateCount; g++) {
      width = Math.max(width, structure.transitionsOf(lists.gateActions.get(g)));
    }
    gateWidth = width;
    long[] gateTransition = new long[gateCount * gateWidth];
    float[] gateMask = new float[gateCount * gateWidth];
    for (int g = 0; g < gateCount; g++) {
      structure.spread(lists.gateActions.get(g), gateTransition, gateMask, g * gateWidth);
    }
    gateActions = staging.addLongs(lists.gateActions);
    gateKinds = staging.addLongs(lists.gateKinds);
    gateRows = staging.addLongs(lists.gateRows);
    gateTransitions =
        gateCount == 0 ? null : staging.add(gateTransition, new Shape(gateTransition.length));
    gateTransitionMask =
        gateCount == 0 ? null : staging.add(gateMask, new Shape(gateCount, gateWidth));
    gateBaseTransitions = staging.addLongs(lists.gateBaseTransitions);
    gateUsesBar =
        gateCount == 0
            ? null
            : staging.add(toBooleans(lists.gateUsesBar), new Shape(gateCount, 1));

    // ē: 行ごとの代表の打牌
    turnCount = lists.turnActions.size();
    int bar = 0;
    int[] representativeDiscards = new int[rows];
    for (int a = 0; a < actions; a++) {
      if (structure.role[a] == ActionRole.DISCARD && structure.representative[a] == a) {
        bar = Math.max(bar, ++representativeDiscards[structure.row[a]]);
      }
    }
    for (int index = 0; index < turnCount; index++) {
      if (representativeDiscards[lists.turnRows.get(index)] == 0) {
        throw new IllegalArgumentException("turn gate in a row without discards");
      }
    }
    barWidth = turnCount == 0 ? 0 : bar;
    if (barWidth > 0) {
      long[] barTransition = new long[rows * barWidth];
      float[] barSlot = new float[rows * barWidth];
      int[] filled = new int[rows];
      for (int a = 0; a < actions; a++) {
        if (structure.role[a] == ActionRole.DISCARD && structure.representative[a] == a) {
          int r = structure.row[a];
          int at = r * barWidth + filled[r]++;
          barTransition[at] = structure.firstTransition(a);
          barSlot[at] = 1f;
        }
      }
      barTransitions = staging.add(barTransition, new Shape(barTransition.length));
      barMask = staging.add(barSlot, new Shape(rows, barWidth));
    } else {
      barTransitions = null;
      barMask = null;
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
    int candidateSpan = 1;
    for (int action : candidates) {
      Objects.checkIndex(action, actions);
      if (!TargetSchema.hasCandidateTargets(structure.role[action])) {
        throw new IllegalArgumentException(
            "action " + action + " (" + structure.role[action] + ") has no candidate targets");
      }
      candidateSpan = Math.max(candidateSpan, structure.transitionsOf(action));
    }
    candidateWidth = candidateSpan;
    if (candidateCount > 0) {
      long[] candidateTransition = new long[candidateCount * candidateWidth];
      float[] candidateSlot = new float[candidateCount * candidateWidth];
      long[] candidateRow = new long[candidateCount];
      for (int c = 0; c < candidateCount; c++) {
        structure.spread(candidates[c], candidateTransition, candidateSlot, c * candidateWidth);
        candidateRow[c] = structure.row[candidates[c]];
      }
      candidateTransitions =
          staging.add(candidateTransition, new Shape(candidateTransition.length));
      candidateMask = staging.add(candidateSlot, new Shape(candidateCount, candidateWidth));
      candidateRows = staging.add(candidateRow, new Shape(candidateCount));
    } else {
      candidateTransitions = null;
      candidateMask = null;
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

    /** 手の代表の遷移を、幅をそろえた列に書く（詰め物は遷移 0・印 0）。 */
    void spread(int action, long[] transitions, float[] mask, int from) {
      int first = firstTransition(action);
      int count = transitionsOf(action);
      for (int t = 0; t < count; t++) {
        transitions[from + t] = first + t;
        mask[from + t] = 1f;
      }
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

    final List<Integer> discardActions = new ArrayList<>();
    final List<Integer> discardTransitions = new ArrayList<>();
    final List<Integer> gateActions = new ArrayList<>();
    final List<Integer> gateKinds = new ArrayList<>();
    final List<Integer> gateRows = new ArrayList<>();
    final List<Integer> gateBaseTransitions = new ArrayList<>();
    final List<Boolean> gateUsesBar = new ArrayList<>();
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
      gateBaseTransitions.add(usesBar ? 0 : structure.firstTransition(base));
      gateUsesBar.add(usesBar);
    }
  }

}
