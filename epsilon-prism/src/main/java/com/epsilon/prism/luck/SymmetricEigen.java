package com.epsilon.prism.luck;

import java.util.Arrays;

/** 実対称行列の固有値と直交固有ベクトルを求める。 */
final class SymmetricEigen {
  private SymmetricEigen() {}

  /** 列優先の上三角を固有ベクトルの列で置き換え、対応する固有値を昇順に返す。 */
  static double[] decompose(double[] matrix, int n) {
    double[] lower = new double[matrix.length];
    for (int j = 0; j < n; j++) {
      for (int i = j; i < n; i++) lower[i + j * n] = matrix[j + i * n];
    }
    double[] diagonal = new double[n], off = new double[n], tau = new double[n];
    tridiagonalize(lower, n, off, tau);
    for (int i = 0; i < n; i++) diagonal[i] = lower[i + i * n];
    Arrays.fill(matrix, 0);
    for (int i = 0; i < n; i++) matrix[i + i * n] = 1;
    for (int k = n - 2; k >= 0; k--) {
      if (tau[k] == 0) continue;
      int column = k * n;
      for (int j = k + 1; j < n; j++) {
        double projection = 0;
        for (int i = k + 1; i < n; i++) projection += lower[column + i] * matrix[j * n + i];
        projection *= tau[k];
        for (int i = k + 1; i < n; i++) matrix[j * n + i] -= projection * lower[column + i];
      }
    }
    diagonalize(diagonal, off, matrix);
    for (int i = 0; i < n; i++) {
      int min = i;
      for (int j = i + 1; j < n; j++) if (diagonal[j] < diagonal[min]) min = j;
      if (min == i) continue;
      double value = diagonal[i];
      diagonal[i] = diagonal[min];
      diagonal[min] = value;
      for (int k = 0; k < n; k++) {
        value = matrix[k + i * n];
        matrix[k + i * n] = matrix[k + min * n];
        matrix[k + min * n] = value;
      }
    }
    return diagonal;
  }

  /** 三重対角行列の副対角を取り出し、各列の対角より下に Householder 反射のベクトルを残す。 */
  private static void tridiagonalize(double[] lower, int n, double[] off, double[] tau) {
    double[] work = new double[n];
    for (int k = 0; k + 1 < n; k++) {
      int first = k + 1, column = k * n;
      double head = lower[column + first], tail = 0;
      for (int i = first + 1; i < n; i++) tail = Math.hypot(tail, lower[column + i]);
      if (tail == 0) {
        off[k] = head;
        continue;
      }
      off[k] = -Math.copySign(Math.hypot(head, tail), head);
      tau[k] = 1 - head / off[k];
      for (int i = first + 1; i < n; i++) lower[column + i] /= head - off[k];
      lower[column + first] = 1;
      Arrays.fill(work, first, n, 0);
      for (int j = first; j < n; j++) {
        double vj = lower[column + j];
        double product = lower[j * n + j] * vj;
        for (int i = j + 1; i < n; i++) {
          double entry = lower[j * n + i];
          work[i] += entry * vj;
          product += entry * lower[column + i];
        }
        work[j] += product;
      }
      double inner = 0;
      for (int i = first; i < n; i++) {
        work[i] *= tau[k];
        inner += work[i] * lower[column + i];
      }
      double adjust = 0.5 * tau[k] * inner;
      for (int i = first; i < n; i++) work[i] -= adjust * lower[column + i];
      for (int j = first; j < n; j++) {
        double vj = lower[column + j], wj = work[j];
        for (int i = j; i < n; i++) lower[j * n + i] -= vj * work[i] + wj * lower[column + i];
      }
    }
  }

  /** 三重対角行列に陰的シフト付き QL 法を適用し、列に固有ベクトルを累積する。 */
  private static void diagonalize(double[] diagonal, double[] off, double[] vectors) {
    int n = diagonal.length;
    for (int first = 0; first < n; first++) {
      for (int iteration = 0; ; iteration++) {
        int end = first;
        while (end + 1 < n
            && Math.abs(off[end])
                > Math.ulp(1d) * (Math.abs(diagonal[end]) + Math.abs(diagonal[end + 1]))) end++;
        if (end == first) break;
        if (iteration >= Double.SIZE * n) {
          throw new IllegalStateException("Ledger eigendecomposition did not converge");
        }
        double ratio = (diagonal[first + 1] - diagonal[first]) / (2 * off[first]);
        double shift = off[first] / (ratio + Math.copySign(Math.hypot(ratio, 1), ratio));
        double g = diagonal[end] - diagonal[first] + shift;
        double sine = 1, cosine = 1, change = 0;
        boolean split = false;
        for (int i = end - 1; i >= first; i--) {
          double f = sine * off[i], b = cosine * off[i];
          double norm = Math.hypot(g, f);
          off[i + 1] = norm;
          if (norm == 0) {
            diagonal[i + 1] -= change;
            split = true;
            break;
          }
          sine = f / norm;
          cosine = g / norm;
          g = diagonal[i + 1] - change;
          double step = (diagonal[i] - g) * sine + 2 * cosine * b;
          change = sine * step;
          diagonal[i + 1] = g + change;
          g = cosine * step - b;
          for (int j = 0; j < n; j++) {
            int col = i * n + j;
            double left = vectors[col], right = vectors[col + n];
            vectors[col] = cosine * left - sine * right;
            vectors[col + n] = sine * left + cosine * right;
          }
        }
        off[end] = 0;
        if (!split) {
          diagonal[first] -= change;
          off[first] = g;
        }
      }
    }
  }
}
