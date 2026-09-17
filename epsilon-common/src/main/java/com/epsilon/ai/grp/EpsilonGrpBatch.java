package com.epsilon.ai.grp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.ArrayList;
import java.util.List;

/** 長さの異なるGRP入力系列にパディングを加え、まとめて計算できる形にしたバッチ。 */
final class EpsilonGrpBatch {

  final int size;
  final int maxSteps;
  final float[] sequences;
  final float[] lengths;
  final float[] labels;
  final int[] finalRanksCodes;

  private EpsilonGrpBatch(
      int size,
      int maxSteps,
      float[] sequences,
      float[] lengths,
      float[] labels,
      int[] finalRanksCodes) {
    this.size = size;
    this.maxSteps = maxSteps;
    this.sequences = sequences;
    this.lengths = lengths;
    this.labels = labels;
    this.finalRanksCodes = finalRanksCodes;
  }

  static EpsilonGrpBatch fromExamples(List<EpsilonGrpExample> examples, int start, int count) {
    List<float[]> sequences = new ArrayList<>(count);
    int[] finalRanksCodes = new int[count];
    int maxSteps = 0;
    for (int i = 0; i < count; i++) {
      EpsilonGrpExample example = examples.get(start + i);
      sequences.add(example.sequence());
      finalRanksCodes[i] = example.finalRanksCode();
      maxSteps = Math.max(maxSteps, EpsilonGrpFeature.steps(example.sequence()));
    }
    return fromTrustedSequences(sequences, finalRanksCodes, maxSteps);
  }

  static EpsilonGrpBatch fromSequences(List<float[]> sequences) {
    List<float[]> validated = new ArrayList<>(sequences.size());
    int maxSteps = 0;
    for (float[] sequence : sequences) {
      float[] copy = EpsilonGrpFeature.validatedCopy(sequence);
      validated.add(copy);
      maxSteps = Math.max(maxSteps, EpsilonGrpFeature.steps(copy));
    }
    int[] labels = new int[validated.size()];
    for (int i = 0; i < labels.length; i++) {
      labels[i] = -1;
    }
    return fromTrustedSequences(validated, labels, maxSteps);
  }

  private static EpsilonGrpBatch fromTrustedSequences(
      List<float[]> source, int[] finalRanksCodes, int maxSteps) {
    int size = source.size();
    float[] sequenceTensor = new float[size * maxSteps * EpsilonGrpFeature.FEATURE_SIZE];
    float[] lengths = new float[size];
    float[] labels = new float[size * EpsilonGrpRanks.MATRIX_SIZE];
    for (int i = 0; i < size; i++) {
      float[] flattened = source.get(i);
      int steps = EpsilonGrpFeature.steps(flattened);
      lengths[i] = steps;
      System.arraycopy(
          flattened,
          0,
          sequenceTensor,
          i * maxSteps * EpsilonGrpFeature.FEATURE_SIZE,
          flattened.length);
      int finalRanksCode = finalRanksCodes[i];
      if (EpsilonGrpRanks.isValidCode(finalRanksCode)) {
        float[] target = EpsilonGrpRanks.oneHotMarginals(finalRanksCode);
        System.arraycopy(
            target, 0, labels, i * EpsilonGrpRanks.MATRIX_SIZE, EpsilonGrpRanks.MATRIX_SIZE);
      }
    }
    return new EpsilonGrpBatch(
        size, maxSteps, sequenceTensor, lengths, labels, finalRanksCodes.clone());
  }

  NDArray sequenceArray(NDManager manager) {
    return manager.create(sequences, new Shape(size, maxSteps, EpsilonGrpFeature.FEATURE_SIZE));
  }

  NDArray lengthArray(NDManager manager) {
    return manager.create(lengths, new Shape(size));
  }

  NDArray labelArray(NDManager manager) {
    return manager.create(labels, new Shape(size, 4, 4));
  }
}
