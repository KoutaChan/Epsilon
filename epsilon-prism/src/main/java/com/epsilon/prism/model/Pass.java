package com.epsilon.prism.model;

import java.util.Objects;

/**
 * 順伝播の種類。推論では錨の勾配も補助の頭も扱わない。学習では C の印で錨の勾配を決め、教師があれば補助の頭を計算する。
 * どちらでも s の値は同じ。GPU の学習では、背骨の注意を自前の HIP の演算にする記録（{@link AttentionTape}）と、当てはめの計測の印
 * （{@link FitProbe}）を渡せる。
 *
 * @param precision 計算の型
 * @param measured 学習の C の印（塊の手の順。推論では null）
 * @param auxiliary 補助の頭の教師（補助の頭を計算しないなら null）
 * @param tape 背骨の注意の自前の演算（記録する形は学習だけ、推論は {@link AttentionTape#forwardOnly}。null なら djl-rocm の演算）
 * @param probe 段の時計の印（学習だけ。計測しないなら {@link FitProbe#NONE}）
 */
public record Pass(
    Precision precision,
    boolean[] measured,
    AuxiliaryLabels auxiliary,
    AttentionTape tape,
    FitProbe probe) {

  /**
   * 組み合わせを確かめる。
   *
   * @throws IllegalArgumentException 推論に補助の教師か記録する注意か計測の印を渡したとき
   */
  public Pass {
    Objects.requireNonNull(precision, "precision");
    Objects.requireNonNull(probe, "probe");
    if (measured == null && auxiliary != null) {
      throw new IllegalArgumentException("auxiliary heads are computed only in training");
    }
    if (measured == null && tape != null && tape.recording()) {
      throw new IllegalArgumentException("a recording attention tape is used only in training");
    }
    if (measured == null && probe != FitProbe.NONE) {
      throw new IllegalArgumentException("a fit probe is used only in training");
    }
  }

  public static Pass inference(Precision precision) {
    return inference(precision, null);
  }

  /**
   * 背骨の注意を自前の演算にする推論の順伝播。
   *
   * @param precision 計算の型
   * @param attention 記録しない注意（{@link AttentionTape#forwardOnly}。null なら djl-rocm の演算）
   * @return 種類
   */
  public static Pass inference(Precision precision, AttentionTape attention) {
    return new Pass(precision, null, null, attention, FitProbe.NONE);
  }

  /**
   * 学習の順伝播。
   *
   * @param precision 計算の型
   * @param measured C の印（塊の手の順）
   * @param auxiliary 補助の頭の教師（null なら補助の頭を計算しない）
   * @return 種類
   */
  public static Pass training(Precision precision, boolean[] measured, AuxiliaryLabels auxiliary) {
    return training(precision, measured, auxiliary, null);
  }

  /**
   * GPU の学習の順伝播（背骨の注意を自前の HIP の演算にする）。
   *
   * @param precision 計算の型
   * @param measured C の印（塊の手の順）
   * @param auxiliary 補助の頭の教師（null なら補助の頭を計算しない）
   * @param tape 注意の記録（逆伝播は {@link AttentionTape#backward} で行う。null なら djl-rocm の演算）
   * @return 種類
   */
  public static Pass training(
      Precision precision, boolean[] measured, AuxiliaryLabels auxiliary, AttentionTape tape) {
    return training(precision, measured, auxiliary, tape, FitProbe.NONE);
  }

  /**
   * 段に時計の印を付ける学習の順伝播（当てはめの学習器）。
   *
   * @param precision 計算の型
   * @param measured C の印（塊の手の順）
   * @param auxiliary 補助の頭の教師（null なら補助の頭を計算しない）
   * @param tape 注意の記録（null なら djl-rocm の演算）
   * @param probe 段の時計の印（計測しないなら {@link FitProbe#NONE}）
   * @return 種類
   */
  public static Pass training(
      Precision precision,
      boolean[] measured,
      AuxiliaryLabels auxiliary,
      AttentionTape tape,
      FitProbe probe) {
    return new Pass(precision, Objects.requireNonNull(measured, "measured"), auxiliary, tape, probe);
  }

  public boolean training() {
    return measured != null;
  }
}
