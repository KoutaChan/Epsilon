package com.epsilon.nano.ai.decision.policy;

import com.epsilon.core.Action;
import com.epsilon.nano.ai.decision.input.DecisionFeatureCodec;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionHostInputs;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import java.util.Arrays;

/**
 * 推論用の未正規化スコアを、方策グラフと同じ階層に従ってCPU上で合法手の対数確率へ合成する。
 *
 * <p>学習では{@link EpsilonDecisionPolicyGraph}のNDArray演算を使って選択経路へ勾配を流す。一方、推論では行動 メタデータが既にCPU上の{@link
 * DecisionHostBatch}に存在するため、同じ小さな条件付き分布をGPU上で再構築する必要がない。この型は
 * スコア計算処理の出力だけを一度同期し、RON/TSUMO、CALL/KAN、種類、具体候補、DAMA/RIICHIの順に同じ確率を合成する。 ネットワーク パラメーターや入力表現は変更しない。
 *
 * <p>同期するスコアの一行レイアウトは {@code [alternative(10), candidate(A), riichiGate(A), valueUtility(0 or 1)]}
 * である。
 */
public final class DecisionPolicyInferenceComposer {

  private static final float ILLEGAL_LOGIT = -1.0e9f;
  private static final int VALUE_SIZE = 1;

  private static final int RON = DecisionFeatureCodec.actionType(Action.Type.RON_AGARI);
  private static final int TSUMO = DecisionFeatureCodec.actionType(Action.Type.TSUMO_AGARI);
  private static final int KYUSHU = DecisionFeatureCodec.actionType(Action.Type.KYUSHU_KYUHAI);
  private static final int PASS = DecisionFeatureCodec.actionType(Action.Type.PASS);
  private static final int[] MELD_TYPE_IDS = encodedTypes(DecisionPolicyNode.meldTypes());
  private static final int[] KAN_TYPE_IDS = encodedTypes(DecisionPolicyNode.kanTypes());
  private static final int DAHAI = DecisionFeatureCodec.actionType(Action.Type.DAHAI);
  private static final int RIICHI_DAHAI = DecisionFeatureCodec.actionType(Action.Type.RIICHI_DAHAI);
  private static final boolean[] RESPONSE_GROUPS = responseGroups();

  private DecisionPolicyInferenceComposer() {}

  /**
   * 一行あたりの同期スコア数を返す。
   *
   * @param legalActionCapacity 連結した行内の候補枠数
   * @param includesValue 末尾へ復号済み期待効用を含めるなら {@code true}
   * @return 連結した行のfloat要素数
   */
  public static int packedScoreWidth(int legalActionCapacity, boolean includesValue) {
    return DecisionAlternative.NETWORK_SIZE
        + legalActionCapacity * 2
        + (includesValue ? VALUE_SIZE : 0);
  }

  /**
   * 連結したスコア計算処理出力を、入力部分ビューと同じ行順の合法手の対数確率へ合成する。
   *
   * @param packedScores {@link #packedScoreWidth}の行を連結したスコア計算処理出力
   * @param hostRows スコア計算処理へ渡したホスト行部分ビュー
   * @param includesValue 連結した行の末尾に復号済み期待効用を含む場合は true
   * @return 形状 {@code [rowCount * legalActionCapacity]} の行優先対数確率
   */
  public static float[] composeLogProbabilities(
      float[] packedScores, DecisionHostBatch.RowSlice hostRows, boolean includesValue) {
    float[] output = composeProbabilities(packedScores, hostRows, includesValue);
    DecisionHostBatch hostBatch = hostRows.source();
    int actionCapacity = hostRows.bucket().legalActionCapacity();
    for (int row = 0; row < hostRows.size(); row++) {
      int actionCount = hostBatch.legalActionCount(hostRows.fromInclusive() + row);
      int outputOffset = row * actionCapacity;
      for (int slot = 0; slot < actionCount; slot++) {
        output[outputOffset + slot] = (float) Math.log(output[outputOffset + slot]);
      }
      Arrays.fill(output, outputOffset + actionCount, outputOffset + actionCapacity, ILLEGAL_LOGIT);
    }
    return output;
  }

  /** 複数の非所有行ビューを、連結した行順のまま一つの対数確率配列へ合成します。 */
  public static float[] composeLogProbabilities(
      float[] packedScores, DecisionHostBatch.RowBatch hostRows, boolean includesValue) {
    float[] output = composeProbabilities(packedScores, hostRows, includesValue);
    int actionCapacity = hostRows.bucket().legalActionCapacity();
    int globalRow = 0;
    for (int part = 0; part < hostRows.sliceCount(); part++) {
      DecisionHostBatch.RowSlice slice = hostRows.slice(part);
      DecisionHostBatch source = slice.source();
      for (int row = 0; row < slice.size(); row++, globalRow++) {
        int actionCount = source.legalActionCount(slice.fromInclusive() + row);
        int outputOffset = globalRow * actionCapacity;
        for (int slot = 0; slot < actionCount; slot++) {
          output[outputOffset + slot] = (float) Math.log(output[outputOffset + slot]);
        }
        Arrays.fill(
            output, outputOffset + actionCount, outputOffset + actionCapacity, ILLEGAL_LOGIT);
      }
    }
    return output;
  }

  /**
   * 連結したスコア計算処理出力を、入力部分ビューと同じ行順の合法手の確率へ合成する。
   *
   * <p>各分岐と条件付きsoftmaxを確率空間で一度だけ合成する。自己対局はこの配列をそのまま無作為抽出へ使い、 ログ 確率へ変換して直後に再度指数関数を適用する往復を行わない。
   *
   * @param packedScores {@link #packedScoreWidth}の行を連結したスコア計算処理出力
   * @param hostRows スコア計算処理へ渡したホスト行部分ビュー
   * @param includesValue 連結した行の末尾に復号済み期待効用を含む場合は true
   * @return 形状 {@code [rowCount * legalActionCapacity]} の行優先確率。パディングは0
   */
  public static float[] composeProbabilities(
      float[] packedScores, DecisionHostBatch.RowSlice hostRows, boolean includesValue) {
    DecisionHostBatch hostBatch = hostRows.source();
    int actionCapacity = hostRows.bucket().legalActionCapacity();
    int packedWidth = packedScoreWidth(actionCapacity, includesValue);
    float[] output = new float[hostRows.size() * actionCapacity];

    for (int row = 0; row < hostRows.size(); row++) {
      int sourceRow = hostRows.fromInclusive() + row;
      composeRowProbabilities(
          packedScores,
          row * packedWidth,
          output,
          row * actionCapacity,
          hostBatch.inputs(),
          sourceRow,
          hostBatch.legalActionCount(sourceRow),
          actionCapacity);
    }
    return output;
  }

  /** 複数の非所有行ビューを中間スコア配列へ分割せず、一つの合法手の確率配列へ合成します。 */
  public static float[] composeProbabilities(
      float[] packedScores, DecisionHostBatch.RowBatch hostRows, boolean includesValue) {
    int actionCapacity = hostRows.bucket().legalActionCapacity();
    int packedWidth = packedScoreWidth(actionCapacity, includesValue);
    float[] output = new float[hostRows.size() * actionCapacity];
    int globalRow = 0;
    for (int part = 0; part < hostRows.sliceCount(); part++) {
      DecisionHostBatch.RowSlice slice = hostRows.slice(part);
      DecisionHostBatch source = slice.source();
      for (int row = 0; row < slice.size(); row++, globalRow++) {
        int sourceRow = slice.fromInclusive() + row;
        composeRowProbabilities(
            packedScores,
            globalRow * packedWidth,
            output,
            globalRow * actionCapacity,
            source.inputs(),
            sourceRow,
            source.legalActionCount(sourceRow),
            actionCapacity);
      }
    }
    return output;
  }

  /**
   * 全候補確率を保持せず、各行で最大方策確率を持つ合法行動候補の位置だけを返す。
   *
   * <p>各行の確率合成は{@link #composeProbabilities(float[], DecisionHostBatch.RowSlice,
   * boolean)}と同じ演算順を使うため、同値確率では先頭枠を選ぶ規則も含めて完全出力で最大確率となる行動と一致する。
   *
   * @param packedScores {@link #packedScoreWidth}の行を連結したスコア計算処理出力
   * @param hostRows スコア計算処理へ渡したホスト行部分ビュー
   * @param includesValue 連結した行の末尾に復号済み期待効用を含む場合は true
   * @return 入力行順の0始まりの合法行動候補の位置
   */
  public static int[] composeGreedyActionSlots(
      float[] packedScores, DecisionHostBatch.RowSlice hostRows, boolean includesValue) {
    DecisionHostBatch hostBatch = hostRows.source();
    DecisionHostInputs inputs = hostBatch.inputs();
    int actionCapacity = hostRows.bucket().legalActionCapacity();
    int packedWidth = packedScoreWidth(actionCapacity, includesValue);
    float[] rowProbabilities = new float[actionCapacity];
    int[] selectedSlots = new int[hostRows.size()];
    for (int row = 0; row < hostRows.size(); row++) {
      int sourceRow = hostRows.fromInclusive() + row;
      int actionCount = hostBatch.legalActionCount(sourceRow);
      Arrays.fill(rowProbabilities, 0, actionCount, 0.0f);
      composeRowProbabilities(
          packedScores,
          row * packedWidth,
          rowProbabilities,
          0,
          inputs,
          sourceRow,
          actionCount,
          actionCapacity);
      int selectedSlot = 0;
      for (int slot = 1; slot < actionCount; slot++) {
        if (rowProbabilities[slot] > rowProbabilities[selectedSlot]) {
          selectedSlot = slot;
        }
      }
      selectedSlots[row] = selectedSlot;
    }
    return selectedSlots;
  }

  /** 複数の非所有行ビューについて、全候補配列を作らず連結した行から最大確率の行動位置を返します。 */
  public static int[] composeGreedyActionSlots(
      float[] packedScores, DecisionHostBatch.RowBatch hostRows, boolean includesValue) {
    int actionCapacity = hostRows.bucket().legalActionCapacity();
    int packedWidth = packedScoreWidth(actionCapacity, includesValue);
    float[] rowProbabilities = new float[actionCapacity];
    int[] selectedSlots = new int[hostRows.size()];
    int globalRow = 0;
    for (int part = 0; part < hostRows.sliceCount(); part++) {
      DecisionHostBatch.RowSlice slice = hostRows.slice(part);
      DecisionHostBatch source = slice.source();
      DecisionHostInputs inputs = source.inputs();
      for (int row = 0; row < slice.size(); row++, globalRow++) {
        int sourceRow = slice.fromInclusive() + row;
        int actionCount = source.legalActionCount(sourceRow);
        Arrays.fill(rowProbabilities, 0, actionCount, 0.0f);
        composeRowProbabilities(
            packedScores,
            globalRow * packedWidth,
            rowProbabilities,
            0,
            inputs,
            sourceRow,
            actionCount,
            actionCapacity);
        int selectedSlot = 0;
        for (int slot = 1; slot < actionCount; slot++) {
          if (rowProbabilities[slot] > rowProbabilities[selectedSlot]) {
            selectedSlot = slot;
          }
        }
        selectedSlots[globalRow] = selectedSlot;
      }
    }
    return selectedSlots;
  }

  private static void composeRowProbabilities(
      float[] packedScores,
      int packedRow,
      float[] output,
      int outputOffset,
      DecisionHostInputs inputs,
      int sourceRow,
      int actionCount,
      int actionCapacity) {
    int candidateOffset = packedRow + DecisionAlternative.NETWORK_SIZE;
    int riichiOffset = candidateOffset + actionCapacity;
    boolean response =
        RESPONSE_GROUPS[inputs.actionCategory(sourceRow, 0, DecisionInputSchema.ActionInt.GROUP)];
    if (response) {
      composeResponse(
          packedScores,
          packedRow,
          candidateOffset,
          output,
          outputOffset,
          inputs,
          sourceRow,
          actionCount);
      return;
    }
    composeTurn(
        packedScores,
        packedRow,
        candidateOffset,
        riichiOffset,
        output,
        outputOffset,
        inputs,
        sourceRow,
        actionCount);
  }

  /** デバイスで復号した期待効用を連結した行の末尾から取り出す。 */
  public static float[] composeValueUtilities(
      float[] packedScores, DecisionHostBatch.RowSlice hostRows) {
    return composeValueUtilities(
        packedScores, hostRows.size(), hostRows.bucket().legalActionCapacity());
  }

  public static float[] composeValueUtilities(
      float[] packedScores, DecisionHostBatch.RowBatch hostRows) {
    return composeValueUtilities(
        packedScores, hostRows.size(), hostRows.bucket().legalActionCapacity());
  }

  private static float[] composeValueUtilities(float[] packedScores, int rows, int actionCapacity) {
    int packedWidth = packedScoreWidth(actionCapacity, true);
    int valueOffset = packedWidth - VALUE_SIZE;
    float[] utilities = new float[rows];
    for (int row = 0; row < rows; row++) {
      utilities[row] = packedScores[row * packedWidth + valueOffset];
    }
    return utilities;
  }

  private static void composeResponse(
      float[] scores,
      int alternativeOffset,
      int candidateOffset,
      float[] output,
      int outputOffset,
      DecisionHostInputs inputs,
      int row,
      int actionCount) {
    long ronSlots = 0L;
    long passSlots = 0L;
    long chiSlots = 0L;
    long ponSlots = 0L;
    long daiminkanSlots = 0L;
    for (int slot = 0; slot < actionCount; slot++) {
      long slotBit = 1L << slot;
      int actionType = type(inputs, row, slot);
      if (actionType == RON) {
        ronSlots |= slotBit;
      } else if (actionType == PASS) {
        passSlots |= slotBit;
      } else if (actionType == MELD_TYPE_IDS[0]) {
        chiSlots |= slotBit;
      } else if (actionType == MELD_TYPE_IDS[1]) {
        ponSlots |= slotBit;
      } else if (actionType == MELD_TYPE_IDS[2]) {
        daiminkanSlots |= slotBit;
      }
    }

    int ronCount = Long.bitCount(ronSlots);
    float unresolvedMass = 1.0f;
    if (ronCount != 0) {
      boolean active = ronCount != actionCount;
      float gate = alternative(scores, alternativeOffset, DecisionAlternative.RON);
      assignCandidates(
          scores, candidateOffset, output, outputOffset, ronSlots, active ? sigmoid(gate) : 1.0f);
      if (active) {
        unresolvedMass = sigmoid(-gate);
      }
    }

    long meldSlots = chiSlots | ponSlots | daiminkanSlots;
    boolean callGateActive = passSlots != 0L && meldSlots != 0L;
    float callGate = alternative(scores, alternativeOffset, DecisionAlternative.CALL);
    if (passSlots != 0L) {
      assignCandidates(
          scores,
          candidateOffset,
          output,
          outputOffset,
          passSlots,
          unresolvedMass * (callGateActive ? sigmoid(-callGate) : 1.0f));
    }
    if (meldSlots == 0L) {
      return;
    }
    float callMass = unresolvedMass * (callGateActive ? sigmoid(callGate) : 1.0f);
    assignAlternativeTypes(
        scores,
        alternativeOffset,
        candidateOffset,
        output,
        outputOffset,
        chiSlots,
        DecisionAlternative.forActionType(DecisionPolicyNode.meldTypes()[0]),
        ponSlots,
        DecisionAlternative.forActionType(DecisionPolicyNode.meldTypes()[1]),
        daiminkanSlots,
        DecisionAlternative.forActionType(DecisionPolicyNode.meldTypes()[2]),
        callMass);
  }

  private static void composeTurn(
      float[] scores,
      int alternativeOffset,
      int candidateOffset,
      int riichiOffset,
      float[] output,
      int outputOffset,
      DecisionHostInputs inputs,
      int row,
      int actionCount) {
    long tsumoSlots = 0L;
    long kyushuSlots = 0L;
    long ankanSlots = 0L;
    long kakanSlots = 0L;
    long discardSlots = 0L;
    long discardIdentitySlots = 0L;
    for (int slot = 0; slot < actionCount; slot++) {
      long slotBit = 1L << slot;
      int actionType = type(inputs, row, slot);
      if (actionType == TSUMO) {
        tsumoSlots |= slotBit;
      } else if (actionType == KYUSHU) {
        kyushuSlots |= slotBit;
      } else if (actionType == KAN_TYPE_IDS[0]) {
        ankanSlots |= slotBit;
      } else if (actionType == KAN_TYPE_IDS[1]) {
        kakanSlots |= slotBit;
      } else if (actionType == DAHAI || actionType == RIICHI_DAHAI) {
        discardSlots |= slotBit;
        if (route(
                inputs, row, slot, DecisionInputSchema.ActionRoute.DISCARD_IDENTITY_REPRESENTATIVE)
            != DecisionInputSchema.PAD_ID) {
          discardIdentitySlots |= slotBit;
        }
      }
    }

    int remaining = actionCount;
    float unresolvedMass = 1.0f;

    int tsumoCount = Long.bitCount(tsumoSlots);
    if (tsumoCount != 0) {
      boolean active = tsumoCount != remaining;
      float gate = alternative(scores, alternativeOffset, DecisionAlternative.TSUMO);
      assignCandidates(
          scores,
          candidateOffset,
          output,
          outputOffset,
          tsumoSlots,
          unresolvedMass * (active ? sigmoid(gate) : 1.0f));
      if (active) {
        unresolvedMass *= sigmoid(-gate);
      }
      remaining -= tsumoCount;
    }

    int kyushuCount = Long.bitCount(kyushuSlots);
    if (kyushuCount != 0) {
      boolean active = kyushuCount != remaining;
      float gate = alternative(scores, alternativeOffset, DecisionAlternative.KYUSHU);
      assignCandidates(
          scores,
          candidateOffset,
          output,
          outputOffset,
          kyushuSlots,
          unresolvedMass * (active ? sigmoid(gate) : 1.0f));
      if (active) {
        unresolvedMass *= sigmoid(-gate);
      }
      remaining -= kyushuCount;
    }

    long kanSlots = ankanSlots | kakanSlots;
    int kanCount = Long.bitCount(kanSlots);
    int discardCount = Long.bitCount(discardSlots);
    boolean kanGateActive = kanCount != 0 && discardCount != 0;
    float kanGate = alternative(scores, alternativeOffset, DecisionAlternative.KAN);
    if (kanCount != 0) {
      assignAlternativeTypes(
          scores,
          alternativeOffset,
          candidateOffset,
          output,
          outputOffset,
          ankanSlots,
          DecisionAlternative.forActionType(DecisionPolicyNode.kanTypes()[0]),
          kakanSlots,
          DecisionAlternative.forActionType(DecisionPolicyNode.kanTypes()[1]),
          unresolvedMass * (kanGateActive ? sigmoid(kanGate) : 1.0f));
    }
    if (kanGateActive) {
      unresolvedMass *= sigmoid(-kanGate);
    }
    if (discardCount != 0) {
      assignDiscards(
          scores,
          candidateOffset,
          riichiOffset,
          output,
          outputOffset,
          inputs,
          row,
          discardSlots,
          discardIdentitySlots,
          unresolvedMass);
    }
  }

  private static void assignAlternativeTypes(
      float[] scores,
      int alternativeOffset,
      int candidateOffset,
      float[] output,
      int outputOffset,
      long firstSlots,
      DecisionAlternative firstAlternative,
      long secondSlots,
      DecisionAlternative secondAlternative,
      float branchMass) {
    assignAlternativeTypes(
        scores,
        alternativeOffset,
        candidateOffset,
        output,
        outputOffset,
        firstSlots,
        firstAlternative,
        secondSlots,
        secondAlternative,
        0L,
        firstAlternative,
        branchMass);
  }

  private static void assignAlternativeTypes(
      float[] scores,
      int alternativeOffset,
      int candidateOffset,
      float[] output,
      int outputOffset,
      long firstSlots,
      DecisionAlternative firstAlternative,
      long secondSlots,
      DecisionAlternative secondAlternative,
      long thirdSlots,
      DecisionAlternative thirdAlternative,
      float branchMass) {
    float max = Float.NEGATIVE_INFINITY;
    if (firstSlots != 0L) {
      max = alternative(scores, alternativeOffset, firstAlternative);
    }
    if (secondSlots != 0L) {
      max = Math.max(max, alternative(scores, alternativeOffset, secondAlternative));
    }
    if (thirdSlots != 0L) {
      max = Math.max(max, alternative(scores, alternativeOffset, thirdAlternative));
    }
    double sum = 0.0;
    if (firstSlots != 0L) {
      sum += Math.exp(alternative(scores, alternativeOffset, firstAlternative) - max);
    }
    if (secondSlots != 0L) {
      sum += Math.exp(alternative(scores, alternativeOffset, secondAlternative) - max);
    }
    if (thirdSlots != 0L) {
      sum += Math.exp(alternative(scores, alternativeOffset, thirdAlternative) - max);
    }
    float inverseNormalizer = (float) (1.0 / sum);
    if (firstSlots != 0L) {
      assignCandidates(
          scores,
          candidateOffset,
          output,
          outputOffset,
          firstSlots,
          branchMass
              * (float) Math.exp(alternative(scores, alternativeOffset, firstAlternative) - max)
              * inverseNormalizer);
    }
    if (secondSlots != 0L) {
      assignCandidates(
          scores,
          candidateOffset,
          output,
          outputOffset,
          secondSlots,
          branchMass
              * (float) Math.exp(alternative(scores, alternativeOffset, secondAlternative) - max)
              * inverseNormalizer);
    }
    if (thirdSlots != 0L) {
      assignCandidates(
          scores,
          candidateOffset,
          output,
          outputOffset,
          thirdSlots,
          branchMass
              * (float) Math.exp(alternative(scores, alternativeOffset, thirdAlternative) - max)
              * inverseNormalizer);
    }
  }

  private static void assignCandidates(
      float[] scores,
      int candidateOffset,
      float[] output,
      int outputOffset,
      long slots,
      float branchMass) {
    float max = Float.NEGATIVE_INFINITY;
    for (long remaining = slots; remaining != 0L; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      max = Math.max(max, scores[candidateOffset + slot]);
    }
    double sum = 0.0;
    for (long remaining = slots; remaining != 0L; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      sum += Math.exp(scores[candidateOffset + slot] - max);
    }
    float scale = branchMass / (float) sum;
    for (long remaining = slots; remaining != 0L; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      output[outputOffset + slot] = (float) Math.exp(scores[candidateOffset + slot] - max) * scale;
    }
  }

  private static void assignDiscards(
      float[] scores,
      int candidateOffset,
      int riichiOffset,
      float[] output,
      int outputOffset,
      DecisionHostInputs inputs,
      int row,
      long discardSlots,
      long discardIdentitySlots,
      float branchMass) {
    float max = Float.NEGATIVE_INFINITY;
    for (long remaining = discardIdentitySlots; remaining != 0L; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      max = Math.max(max, scores[candidateOffset + slot]);
    }
    double sum = 0.0;
    for (long remaining = discardIdentitySlots; remaining != 0L; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      sum += Math.exp(scores[candidateOffset + slot] - max);
    }
    float identityScale = branchMass / (float) sum;
    for (long remaining = discardSlots; remaining != 0L; remaining &= remaining - 1) {
      int slot = Long.numberOfTrailingZeros(remaining);
      int type = type(inputs, row, slot);
      int representativeSlot =
          route(inputs, row, slot, DecisionInputSchema.ActionRoute.DISCARD_REPRESENTATIVE_SLOT) - 1;
      int damaSlot = route(inputs, row, slot, DecisionInputSchema.ActionRoute.DAMA_SLOT) - 1;
      int riichiSlot = route(inputs, row, slot, DecisionInputSchema.ActionRoute.RIICHI_SLOT) - 1;
      float variantProbability = 1.0f;
      if (damaSlot >= 0 && riichiSlot >= 0) {
        float gate = scores[riichiOffset + riichiSlot];
        variantProbability = sigmoid(type == RIICHI_DAHAI ? gate : -gate);
      }
      output[outputOffset + slot] =
          (float) Math.exp(scores[candidateOffset + representativeSlot] - max)
              * identityScale
              * variantProbability;
    }
  }

  private static int type(DecisionHostInputs inputs, int row, int slot) {
    return inputs.actionCategory(row, slot, DecisionInputSchema.ActionInt.TYPE);
  }

  private static int route(
      DecisionHostInputs inputs, int row, int slot, DecisionInputSchema.ActionRoute route) {
    return inputs.actionRoute(row, slot, route);
  }

  private static float alternative(
      float[] scores, int alternativeOffset, DecisionAlternative alternative) {
    return scores[alternativeOffset + alternative.networkIndex()];
  }

  private static float sigmoid(float value) {
    if (value >= 0.0f) {
      double exp = Math.exp(-value);
      return (float) (1.0 / (1.0 + exp));
    }
    double exp = Math.exp(value);
    return (float) (exp / (1.0 + exp));
  }

  private static boolean[] responseGroups() {
    boolean[] responseGroups = new boolean[Action.Group.values().length + 1];
    for (Action.Group group : Action.Group.values()) {
      responseGroups[DecisionFeatureCodec.actionGroup(group)] = group.isResponse();
    }
    return responseGroups;
  }

  private static int[] encodedTypes(Action.Type[] types) {
    int[] encoded = new int[types.length];
    for (int index = 0; index < types.length; index++) {
      encoded[index] = DecisionFeatureCodec.actionType(types[index]);
    }
    return encoded;
  }
}
