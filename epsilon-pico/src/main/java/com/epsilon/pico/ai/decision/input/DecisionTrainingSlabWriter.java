package com.epsilon.pico.ai.decision.input;

import com.epsilon.core.GameState;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;

/**
 * 確定した小バッチの最終入力バッファへ、保存済みの対局データを直接書き込む。
 *
 * <p>保存形式、ヒープまたはページロック済みの記憶領域、デバイス転送を分離するため、プリミティブ型のバッファだけを扱う。1行ずつ保存されたデータをテンソルごとの配置へ並べ替え、方策のパディング部分は行ごとに0で上書きする。
 */
public final class DecisionTrainingSlabWriter {

  private static final int VALUE_TARGET_SIZE = DecisionTrainingTargets.VALUE_TARGET_SIZE;

  private final int rows;
  private final DecisionBucket bucket;
  private final DecisionInputLayout inputLayout;
  private final DecisionTrainingTargetLayout targetLayout;
  private final ShortBuffer inputCategories;
  private final FloatBuffer inputNumerics;
  private final IntBuffer targetCategories;
  private final FloatBuffer targetNumerics;
  private final IntBuffer playerMemoryPresentIndices;
  private final IntBuffer transitionPresentIndices;

  private int inputRows;
  private int targetRows;
  private int pendingInputRow = -1;
  private int pendingTargetRow = -1;
  private int pendingTargetLegalActionCount;
  private int playerMemoryPresentCount;
  private int transitionPresentCount;
  private boolean sealed;

  public DecisionTrainingSlabWriter(
      int rows,
      DecisionBucket bucket,
      ShortBuffer inputCategories,
      FloatBuffer inputNumerics,
      IntBuffer targetCategories,
      FloatBuffer targetNumerics,
      IntBuffer playerMemoryPresentIndices,
      IntBuffer transitionPresentIndices) {
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be positive: " + rows);
    }
    this.rows = rows;
    this.bucket = java.util.Objects.requireNonNull(bucket, "bucket");
    inputLayout = new DecisionInputLayout(rows, bucket);
    targetLayout = new DecisionTrainingTargetLayout(rows, bucket);
    this.inputCategories =
        exactWritable(inputCategories, inputLayout.categoricalElementCount(), "input categories");
    this.inputNumerics =
        exactWritable(inputNumerics, inputLayout.numericElementCount(), "input numerics");
    this.targetCategories =
        exactWritable(
            targetCategories, targetLayout.categoricalElementCount(), "target categories");
    this.targetNumerics =
        exactWritable(targetNumerics, targetLayout.numericElementCount(), "target numerics");
    this.playerMemoryPresentIndices =
        writableAtLeast(
            playerMemoryPresentIndices,
            maximumPlayerMemoryPresentCount(rows),
            "player-memory indices");
    this.transitionPresentIndices =
        writableAtLeast(
            transitionPresentIndices,
            Math.multiplyExact(
                rows,
                Math.multiplyExact(
                    bucket.legalActionCapacity(), bucket.actionTransitionCapacity())),
            "transition indices");
  }

  /** 一行入力と確率分布の本体の二本の入力連続バッファを最終バッチの指定行へ指定位置へ配置する。 */
  public void writeInputRow(
      int row,
      DecisionInputLayout encodedLayout,
      ShortBuffer encodedCategories,
      FloatBuffer encodedNumerics) {
    writeInputCategories(row, encodedLayout, encodedCategories);
    writeInputNumerics(row, encodedLayout, encodedNumerics);
  }

  /** 一行入力と確率分布の本体のカテゴリ値連続バッファを指定位置へ配置する。直後に同じ行の数値連続バッファを書く。 */
  public void writeInputCategories(
      int row, DecisionInputLayout encodedLayout, ShortBuffer encodedCategories) {
    requireOpen();
    requireNextRow(row, inputRows, "input");
    requireEncodedLayout(encodedLayout);
    if (pendingInputRow != -1) {
      throw new IllegalStateException("a categorical input row is already pending");
    }
    ShortBuffer categories =
        exactReadable(
            encodedCategories, encodedLayout.categoricalElementCount(), "encoded input categories");
    for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
      if (tensor.slab() != DecisionInputLayout.Slab.CATEGORICAL) {
        continue;
      }
      DecisionInputLayout.Region source = encodedLayout.region(tensor);
      DecisionInputLayout.Region destination = inputLayout.region(tensor);
      copy(
          categories,
          source.slabOffset(),
          inputCategories,
          destination.rowOffset(row),
          source.elementsPerRow());
    }
    pendingInputRow = row;
  }

  /** 保留中のカテゴリ値行と同じ行の数値連続バッファを指定位置へ配置し、派生インデックスを確定する。 */
  public void writeInputNumerics(
      int row, DecisionInputLayout encodedLayout, FloatBuffer encodedNumerics) {
    requireOpen();
    requireNextRow(row, inputRows, "input");
    requireEncodedLayout(encodedLayout);
    if (pendingInputRow != row) {
      throw new IllegalStateException(
          "numeric input row has no matching categorical row: row="
              + row
              + " pending="
              + pendingInputRow);
    }
    FloatBuffer numerics =
        exactReadable(
            encodedNumerics, encodedLayout.numericElementCount(), "encoded input numerics");
    for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
      if (tensor.slab() != DecisionInputLayout.Slab.NUMERIC) {
        continue;
      }
      DecisionInputLayout.Region source = encodedLayout.region(tensor);
      DecisionInputLayout.Region destination = inputLayout.region(tensor);
      copy(
          numerics,
          source.slabOffset(),
          inputNumerics,
          destination.rowOffset(row),
          source.elementsPerRow());
    }
    validateActionRow(row);
    appendPlayerMemoryPresentIndices(row);
    appendTransitionPresentIndices(row);
    pendingInputRow = -1;
    inputRows++;
  }

  /** メタデータ教師値と入力と確率分布の本体内の二本の方策を、最終教師値連続バッファへ直接書く。 */
  public void writeTargetRow(
      int row,
      int legalActionCount,
      int chosenSlot,
      FloatBuffer behaviorPolicy,
      FloatBuffer rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    writeBehaviorPolicy(row, legalActionCount, behaviorPolicy, actorWeight > 0.0f);
    writeRemainingTargetRow(
        row,
        legalActionCount,
        chosenSlot,
        rolloutPolicy,
        valueTarget,
        advantage,
        actorWeight,
        sampleWeight);
  }

  /** 実際の行動選択に使う方策を先に最終教師値連続バッファへ書き、同じ符号化・復号処理一時バッファの再利用を可能にする。 */
  public void writeBehaviorPolicy(
      int row, int legalActionCount, FloatBuffer behaviorPolicy, boolean requireFullSupport) {
    requireOpen();
    requireNextRow(row, targetRows, "target");
    if (inputRows != row + 1) {
      throw new IllegalStateException("input row must be written before its target: row=" + row);
    }
    if (pendingTargetRow != -1) {
      throw new IllegalStateException("a behavior Policy row is already pending");
    }
    if (legalActionCount < 1 || legalActionCount > bucket.legalActionCapacity()) {
      throw new IllegalArgumentException("legalActionCount out of range: " + legalActionCount);
    }
    FloatBuffer behavior = exactReadable(behaviorPolicy, legalActionCount, "behavior policy");
    writePolicy(
        behavior,
        targetLayout.behaviorPolicyRowOffset(row),
        legalActionCount,
        requireFullSupport,
        "behaviorPolicy");
    pendingTargetRow = row;
    pendingTargetLegalActionCount = legalActionCount;
  }

  /** 探索前の方策とメタデータ教師値を書き、保留中の教師値行を確定する。 */
  public void writeRemainingTargetRow(
      int row,
      int legalActionCount,
      int chosenSlot,
      FloatBuffer rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    requireOpen();
    requireNextRow(row, targetRows, "target");
    if (pendingTargetRow != row || pendingTargetLegalActionCount != legalActionCount) {
      throw new IllegalStateException(
          "target row has no matching behavior Policy: row="
              + row
              + " pending="
              + pendingTargetRow
              + " legalActions="
              + legalActionCount
              + " pendingLegalActions="
              + pendingTargetLegalActionCount);
    }
    if (chosenSlot < 0 || chosenSlot >= legalActionCount) {
      throw new IllegalArgumentException("chosenSlot outside legal actions: " + chosenSlot);
    }
    FloatBuffer rollout = exactReadable(rolloutPolicy, legalActionCount, "rollout policy");

    requireFiniteNonNegative(actorWeight, "actorWeight");
    requireFiniteNonNegative(sampleWeight, "sampleWeight");

    targetCategories.put(row, chosenSlot);
    writePolicy(
        rollout,
        targetLayout.rolloutPolicyRowOffset(row),
        legalActionCount,
        false,
        "rolloutPolicy");
    int targetOffset = targetLayout.rowNumericOffset(row);
    targetNumerics.put(targetOffset, requireFinite(valueTarget, "valueTarget"));
    targetNumerics.put(targetOffset + VALUE_TARGET_SIZE, requireFinite(advantage, "advantage"));
    targetNumerics.put(targetOffset + targetLayout.actorWeightComponent(), actorWeight);
    targetNumerics.put(targetOffset + targetLayout.sampleWeightComponent(), sampleWeight);
    pendingTargetRow = -1;
    pendingTargetLegalActionCount = 0;
    targetRows++;
  }

  /** 全行が一度ずつ書かれたことを確認し、以後の更新を禁止する。 */
  public void seal() {
    requireOpen();
    if (pendingInputRow != -1
        || pendingTargetRow != -1
        || inputRows != rows
        || targetRows != rows) {
      throw new IllegalStateException(
          "training slab is incomplete: inputRows="
              + inputRows
              + " targetRows="
              + targetRows
              + " pendingInputRow="
              + pendingInputRow
              + " pendingTargetRow="
              + pendingTargetRow
              + " expected="
              + rows);
    }
    sealed = true;
  }

  public int rows() {
    return rows;
  }

  public DecisionBucket bucket() {
    return bucket;
  }

  public DecisionInputLayout inputLayout() {
    return inputLayout;
  }

  public DecisionTrainingTargetLayout targetLayout() {
    return targetLayout;
  }

  public int legalActionCount(int row) {
    requireWrittenInputRow(row);
    return encodedLegalActionCount(row);
  }

  private int encodedLegalActionCount(int row) {
    int count = 0;
    while (count < bucket.legalActionCapacity()
        && encodedActionCategory(row, count, DecisionInputSchema.ActionInt.ID)
            != DecisionInputSchema.PAD_ID) {
      count++;
    }
    return count;
  }

  public int legalActionId(int row, int slot) {
    if (slot < 0 || slot >= legalActionCount(row)) {
      throw new IndexOutOfBoundsException("legal action slot=" + slot);
    }
    return DecisionFeatureCodec.decodeActionId(
        encodedActionCategory(row, slot, DecisionInputSchema.ActionInt.ID));
  }

  public int playerSeat(int row) {
    return roundCategory(row, DecisionInputSchema.RoundInt.PLAYER_SEAT) - 1;
  }

  public int sourcePlayerRelativeSeat(int row) {
    return roundCategory(row, DecisionInputSchema.RoundInt.SOURCE_PLAYER_RELATIVE_SEAT) - 1;
  }

  public int currentPlayerRelativeSeat(int row) {
    return roundCategory(row, DecisionInputSchema.RoundInt.CURRENT_PLAYER_RELATIVE_SEAT) - 1;
  }

  public float actorWeight(int row) {
    requireWrittenTargetRow(row);
    return targetNumerics.get(
        targetLayout.rowNumericOffset(row) + targetLayout.actorWeightComponent());
  }

  public float sampleWeight(int row) {
    requireWrittenTargetRow(row);
    return targetNumerics.get(
        targetLayout.rowNumericOffset(row) + targetLayout.sampleWeightComponent());
  }

  public float advantage(int row) {
    requireWrittenTargetRow(row);
    return targetNumerics.get(targetLayout.rowNumericOffset(row) + VALUE_TARGET_SIZE);
  }

  public int playerMemoryPresentCount() {
    requireSealed();
    return playerMemoryPresentCount;
  }

  public int transitionPresentCount() {
    requireSealed();
    return transitionPresentCount;
  }

  private void appendPlayerMemoryPresentIndices(int row) {
    int tokensPerPlayer =
        1
            + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
            + DecisionInputSchema.MAX_MELDS_PER_PLAYER;
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      int playerOffset = (row * GameState.NUM_PLAYERS + player) * tokensPerPlayer;
      putPlayerMemoryIndex(playerOffset);
      int riverCount = playerCategory(row, player, DecisionInputSchema.PlayerInt.RIVER_COUNT);
      int meldCount = playerCategory(row, player, DecisionInputSchema.PlayerInt.MELD_COUNT);
      if (riverCount < 0 || riverCount > DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER) {
        throw new IllegalArgumentException("river count outside schema: " + riverCount);
      }
      if (meldCount < 0 || meldCount > DecisionInputSchema.MAX_MELDS_PER_PLAYER) {
        throw new IllegalArgumentException("meld count outside schema: " + meldCount);
      }
      for (int river = 0; river < riverCount; river++) {
        putPlayerMemoryIndex(playerOffset + 1 + river);
      }
      int meldOffset = playerOffset + 1 + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER;
      for (int meld = 0; meld < meldCount; meld++) {
        putPlayerMemoryIndex(meldOffset + meld);
      }
    }
  }

  private void appendTransitionPresentIndices(int row) {
    int actionCapacity = bucket.legalActionCapacity();
    int transitionCapacity = bucket.actionTransitionCapacity();
    int transitionsPerRow = Math.multiplyExact(actionCapacity, transitionCapacity);
    int actions = encodedLegalActionCount(row);
    for (int action = 0; action < actions; action++) {
      int transition = 0;
      while (transition < transitionCapacity && transitionPresent(row, action, transition)) {
        transitionPresentIndices.put(
            transitionPresentCount++,
            row * transitionsPerRow + action * transitionCapacity + transition);
        transition++;
      }
      if (transition == 0) {
        throw new IllegalArgumentException(
            "legal action has no transition: row=" + row + " action=" + action);
      }
      for (; transition < transitionCapacity; transition++) {
        if (transitionPresent(row, action, transition)) {
          throw new IllegalArgumentException(
              "transitions are not contiguous: row="
                  + row
                  + " action="
                  + action
                  + " transition="
                  + transition);
        }
      }
    }
  }

  private void validateActionRow(int row) {
    int legalActionCount = encodedLegalActionCount(row);
    if (legalActionCount < 1) {
      throw new IllegalArgumentException("training input row has no legal action: row=" + row);
    }
    for (int action = legalActionCount; action < bucket.legalActionCapacity(); action++) {
      if (encodedActionCategory(row, action, DecisionInputSchema.ActionInt.ID)
          != DecisionInputSchema.PAD_ID) {
        throw new IllegalArgumentException(
            "legal actions are not contiguous: row=" + row + " action=" + action);
      }
    }
    boolean response =
        DecisionFeatureCodec.isResponseActionGroup(
            encodedActionCategory(row, 0, DecisionInputSchema.ActionInt.GROUP));
    for (int action = 1; action < legalActionCount; action++) {
      boolean candidateResponse =
          DecisionFeatureCodec.isResponseActionGroup(
              encodedActionCategory(row, action, DecisionInputSchema.ActionInt.GROUP));
      if (candidateResponse != response) {
        throw new IllegalArgumentException(
            "turn and response actions share a row: row=" + row + " action=" + action);
      }
    }
  }

  private boolean transitionPresent(int row, int action, int transition) {
    DecisionInputLayout.Region region =
        inputLayout.region(DecisionInputLayout.Tensor.TRANSITION_CATEGORIES);
    int flatTransition =
        Math.addExact(Math.multiplyExact(action, bucket.actionTransitionCapacity()), transition);
    int offset =
        Math.addExact(
            region.rowOffset(row),
            Math.addExact(
                Math.multiplyExact(
                    flatTransition, DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE),
                DecisionInputSchema.ActionTransitionInt.PRESENT.ordinal()));
    return inputCategories.get(offset) != DecisionInputSchema.PAD_ID;
  }

  private int roundCategory(int row, DecisionInputSchema.RoundInt field) {
    requireWrittenInputRow(row);
    DecisionInputLayout.Region region =
        inputLayout.region(DecisionInputLayout.Tensor.STATE_CATEGORIES);
    return inputCategories.get(region.rowOffset(row) + field.ordinal());
  }

  private int playerCategory(int row, int player, DecisionInputSchema.PlayerInt field) {
    DecisionInputLayout.Region region =
        inputLayout.region(DecisionInputLayout.Tensor.STATE_CATEGORIES);
    int offset =
        region.rowOffset(row)
            + DecisionInputSchema.ROUND_INT_COUNT
            + player * DecisionInputSchema.PLAYER_INT_STRIDE
            + field.ordinal();
    return inputCategories.get(offset);
  }

  private int encodedActionCategory(int row, int action, DecisionInputSchema.ActionInt field) {
    DecisionInputLayout.Region region =
        inputLayout.region(DecisionInputLayout.Tensor.ACTION_CATEGORIES);
    return inputCategories.get(
        region.rowOffset(row) + action * DecisionInputSchema.ACTION_INT_STRIDE + field.ordinal());
  }

  private void writePolicy(
      FloatBuffer source,
      int destinationOffset,
      int legalActionCount,
      boolean requireFullSupport,
      String label) {
    float sum = 0.0f;
    for (int slot = 0; slot < legalActionCount; slot++) {
      float value = source.get(slot);
      if (!Float.isFinite(value) || value < 0.0f || (requireFullSupport && value == 0.0f)) {
        throw new IllegalArgumentException(label + " has invalid probability at slot " + slot);
      }
      targetNumerics.put(destinationOffset + slot, value);
      sum += value;
    }
    if (Math.abs(sum - 1.0f) > 1.0e-4f) {
      throw new IllegalArgumentException(label + " is not normalized: " + sum);
    }
    for (int slot = legalActionCount; slot < bucket.legalActionCapacity(); slot++) {
      targetNumerics.put(destinationOffset + slot, 0.0f);
    }
  }

  private void putPlayerMemoryIndex(int value) {
    playerMemoryPresentIndices.put(playerMemoryPresentCount++, value);
  }

  private static int maximumPlayerMemoryPresentCount(int rows) {
    int tokensPerPlayer =
        1
            + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
            + DecisionInputSchema.MAX_MELDS_PER_PLAYER;
    return Math.multiplyExact(rows, Math.multiplyExact(GameState.NUM_PLAYERS, tokensPerPlayer));
  }

  private void requireWrittenInputRow(int row) {
    if (row < 0 || row >= inputRows) {
      throw new IndexOutOfBoundsException("input row=" + row + " written=" + inputRows);
    }
  }

  private void requireWrittenTargetRow(int row) {
    if (row < 0 || row >= targetRows) {
      throw new IndexOutOfBoundsException("target row=" + row + " written=" + targetRows);
    }
  }

  private void requireOpen() {
    if (sealed) {
      throw new IllegalStateException("training slab is already sealed");
    }
  }

  private void requireSealed() {
    if (!sealed) {
      throw new IllegalStateException("training slab is not sealed");
    }
  }

  private void requireNextRow(int row, int expected, String label) {
    if (row != expected || row >= rows) {
      throw new IllegalStateException(
          label + " rows must be written in order: expected=" + expected + " actual=" + row);
    }
  }

  private void requireEncodedLayout(DecisionInputLayout encodedLayout) {
    if (encodedLayout.capacity() != 1 || !encodedLayout.bucket().equals(bucket)) {
      throw new IllegalArgumentException(
          "encoded input layout mismatch: expected bucket="
              + bucket
              + " actual="
              + encodedLayout.bucket()
              + " capacity="
              + encodedLayout.capacity());
    }
  }

  private static void copy(
      ShortBuffer source,
      int sourceOffset,
      ShortBuffer destination,
      int destinationOffset,
      int length) {
    ShortBuffer src = source.duplicate();
    src.position(sourceOffset).limit(Math.addExact(sourceOffset, length));
    ShortBuffer dst = destination.duplicate();
    dst.position(destinationOffset).limit(Math.addExact(destinationOffset, length));
    dst.put(src);
  }

  private static void copy(
      FloatBuffer source,
      int sourceOffset,
      FloatBuffer destination,
      int destinationOffset,
      int length) {
    FloatBuffer src = source.duplicate();
    src.position(sourceOffset).limit(Math.addExact(sourceOffset, length));
    FloatBuffer dst = destination.duplicate();
    dst.position(destinationOffset).limit(Math.addExact(destinationOffset, length));
    dst.put(src);
  }

  private static ShortBuffer exactWritable(ShortBuffer buffer, int elements, String label) {
    ShortBuffer view = java.util.Objects.requireNonNull(buffer, label).duplicate();
    view.clear();
    if (view.capacity() != elements || view.isReadOnly()) {
      throw new IllegalArgumentException(
          label + " capacity must be exactly " + elements + ": " + view.capacity());
    }
    return view;
  }

  private static FloatBuffer exactWritable(FloatBuffer buffer, int elements, String label) {
    FloatBuffer view = java.util.Objects.requireNonNull(buffer, label).duplicate();
    view.clear();
    if (view.capacity() != elements || view.isReadOnly()) {
      throw new IllegalArgumentException(
          label + " capacity must be exactly " + elements + ": " + view.capacity());
    }
    return view;
  }

  private static IntBuffer exactWritable(IntBuffer buffer, int elements, String label) {
    IntBuffer view = java.util.Objects.requireNonNull(buffer, label).duplicate();
    view.clear();
    if (view.capacity() != elements || view.isReadOnly()) {
      throw new IllegalArgumentException(
          label + " capacity must be exactly " + elements + ": " + view.capacity());
    }
    return view;
  }

  private static IntBuffer writableAtLeast(IntBuffer buffer, int elements, String label) {
    IntBuffer view = java.util.Objects.requireNonNull(buffer, label).duplicate();
    view.clear();
    if (view.capacity() < elements || view.isReadOnly()) {
      throw new IllegalArgumentException(
          label + " capacity must be at least " + elements + ": " + view.capacity());
    }
    return view;
  }

  private static ShortBuffer exactReadable(ShortBuffer buffer, int elements, String label) {
    ShortBuffer view = java.util.Objects.requireNonNull(buffer, label).duplicate();
    if (view.remaining() != elements) {
      throw new IllegalArgumentException(
          label + " remaining must be " + elements + ": " + view.remaining());
    }
    return view.slice();
  }

  private static FloatBuffer exactReadable(FloatBuffer buffer, int elements, String label) {
    FloatBuffer view = java.util.Objects.requireNonNull(buffer, label).duplicate();
    if (view.remaining() != elements) {
      throw new IllegalArgumentException(
          label + " remaining must be " + elements + ": " + view.remaining());
    }
    return view.slice();
  }

  private static float requireFinite(float value, String label) {
    if (!Float.isFinite(value)) {
      throw new IllegalArgumentException(label + " must be finite: " + value);
    }
    return value;
  }

  private static void requireFiniteNonNegative(float value, String label) {
    if (!Float.isFinite(value) || value < 0.0f) {
      throw new IllegalArgumentException(label + " must be finite and non-negative: " + value);
    }
  }
}
