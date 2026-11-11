package com.epsilon.prism.model;

/**
 * 1 回の学習で使う学習率のスケジュール。最初の数回の更新で 0 からピークまで線形に上げ、その後 cosine で下限まで下げる。更新の番号は、その学習の
 * 最初の更新を 1 とする番号で、反復をまたいで通して数えない。
 *
 * @param peak 学習率のピーク
 * @param floor 学習率の下限
 * @param warmup 学習率を上げていく更新の回数
 * @param steps その学習の更新の回数（最後の更新で下限になる）
 */
public record LearningRateSchedule(double peak, double floor, int warmup, int steps) {
  /** 学習率を上げていく回数を、更新の回数に割合を掛けた値（切り上げ、1 以上）にして作る。 */
  public static LearningRateSchedule of(double peak, double floor, double fraction, int steps) {
    return new LearningRateSchedule(
        peak, floor, Math.max(1, (int) Math.ceil(fraction * steps)), steps);
  }

  /** 指定した回の更新の学習率を返す。回は 1 から数える。 */
  public double rate(int step) {
    if (step <= warmup) return peak * step / warmup;
    double progress = (step - warmup) / (double) (steps - warmup);
    return floor + (peak - floor) * 0.5 * (1 + Math.cos(Math.PI * progress));
  }
}
