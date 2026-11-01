package com.epsilon.prism.model;

/**
 * 1回の当てはめの学習率。始めの warmup 回で 0 から peak まで線形に上げ、そのあと cosine で floor まで下げる。番号は当てはめの最初の更新を 1
 * とする相対の番号で、反復をまたいで通算しない。
 *
 * @param steps 当てはめの更新の数（最後の更新で floor）
 */
public record LearningRateSchedule(double peak, double floor, int warmup, int steps) {
  /** warmup を steps の fraction 倍（切り上げ、1 以上）にする。 */
  public static LearningRateSchedule of(double peak, double floor, double fraction, int steps) {
    return new LearningRateSchedule(
        peak, floor, Math.max(1, (int) Math.ceil(fraction * steps)), steps);
  }

  /** step 回目（1 から）の更新の学習率。 */
  public double rate(int step) {
    if (step <= warmup) return peak * step / warmup;
    double progress = (step - warmup) / (double) (steps - warmup);
    return floor + (peak - floor) * 0.5 * (1 + Math.cos(Math.PI * progress));
  }
}
