package com.epsilon.pico.ai.decision.runtime;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import java.util.ArrayList;
import java.util.List;

/** デバイス出力の連結と、ホスト側の行ビューへの復号を一元化する。 */
final class DecisionInferenceOutputCodec {

  private DecisionInferenceOutputCodec() {}

  static NDArray packFloat32Scores(NDList scores) {
    return NDArrays.concat(scores, 1).toType(DataType.FLOAT32, false);
  }

  static List<EpsilonDecisionInferenceServer.Prediction> decodePredictions(
      DecisionHostBatch.RowSlice hostRows,
      float[] policyValues,
      boolean policyValuesAreProbabilities,
      float[] valueUtilities) {
    DecisionHostBatch hostBatch = hostRows.source();
    int legalActionCapacity = hostBatch.bucket().legalActionCapacity();
    boolean includesValue = valueUtilities.length != 0;
    ArrayList<EpsilonDecisionInferenceServer.Prediction> predictions =
        new ArrayList<>(hostRows.size());
    for (int rowIndex = 0; rowIndex < hostRows.size(); rowIndex++) {
      int sourceRow = hostRows.fromInclusive() + rowIndex;
      int legalActionCount = hostBatch.legalActionCount(sourceRow);
      int policyOffset = rowIndex * legalActionCapacity;
      float valueUtility = includesValue ? valueUtilities[rowIndex] : Float.NaN;
      predictions.add(
          policyValuesAreProbabilities
              ? EpsilonDecisionInferenceServer.Prediction.probabilityView(
                  policyValues, policyOffset, legalActionCount, valueUtility)
              : EpsilonDecisionInferenceServer.Prediction.view(
                  policyValues, policyOffset, legalActionCount, valueUtility));
    }
    return predictions;
  }

  /** 複数の非所有行ビューを、共有する最終出力配列へのビューとして復号します。 */
  static List<EpsilonDecisionInferenceServer.Prediction> decodePredictions(
      DecisionHostBatch.RowBatch hostRows,
      float[] policyValues,
      boolean policyValuesAreProbabilities,
      float[] valueUtilities) {
    int legalActionCapacity = hostRows.bucket().legalActionCapacity();
    boolean includesValue = valueUtilities.length != 0;
    ArrayList<EpsilonDecisionInferenceServer.Prediction> predictions =
        new ArrayList<>(hostRows.size());
    int globalRow = 0;
    for (int part = 0; part < hostRows.sliceCount(); part++) {
      DecisionHostBatch.RowSlice slice = hostRows.slice(part);
      DecisionHostBatch source = slice.source();
      for (int row = 0; row < slice.size(); row++, globalRow++) {
        int legalActionCount = source.legalActionCount(slice.fromInclusive() + row);
        int policyOffset = globalRow * legalActionCapacity;
        float valueUtility = includesValue ? valueUtilities[globalRow] : Float.NaN;
        predictions.add(
            policyValuesAreProbabilities
                ? EpsilonDecisionInferenceServer.Prediction.probabilityView(
                    policyValues, policyOffset, legalActionCount, valueUtility)
                : EpsilonDecisionInferenceServer.Prediction.view(
                    policyValues, policyOffset, legalActionCount, valueUtility));
      }
    }
    return predictions;
  }

  static List<EpsilonDecisionInferenceServer.Prediction> forcedActions(int rowCount) {
    ArrayList<EpsilonDecisionInferenceServer.Prediction> predictions = new ArrayList<>(rowCount);
    for (int row = 0; row < rowCount; row++) {
      predictions.add(EpsilonDecisionInferenceServer.Prediction.forcedAction());
    }
    return predictions;
  }

  static List<EpsilonDecisionInferenceServer.Prediction> forcedActionsWithValues(
      float[] valueUtilities, int rowCount) {
    ArrayList<EpsilonDecisionInferenceServer.Prediction> predictions = new ArrayList<>(rowCount);
    for (int row = 0; row < rowCount; row++) {
      predictions.add(
          EpsilonDecisionInferenceServer.Prediction.forcedActionWithValue(valueUtilities[row]));
    }
    return predictions;
  }

  static float[] concatenate(List<float[]> parts) {
    int size = 0;
    for (float[] part : parts) {
      size += part.length;
    }
    float[] result = new float[size];
    int offset = 0;
    for (float[] part : parts) {
      System.arraycopy(part, 0, result, offset, part.length);
      offset += part.length;
    }
    return result;
  }
}
