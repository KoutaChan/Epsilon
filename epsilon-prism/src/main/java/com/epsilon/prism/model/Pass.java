package com.epsilon.prism.model;

import com.epsilon.agent.runtime.Precision;
import java.util.Objects;

/**
 * 順伝播の種類。推論ではアンカーの勾配も補助のヘッドも扱わない。学習では、測った手の集合 C でアンカーの勾配を決め、教師があれば
 * 補助のヘッドを計算する。どちらでも s の値は変わらない。GPU の学習では、バックボーンの Attention を自前の HIP の演算で行う記録
 * （{@link AttentionTape}）と、学習の計測点（{@link FitProbe}）を渡せる。
 *
 * @param precision 計算の型
 * @param measured 学習で測った手 C の印（バッチの中の手の順。推論では null）
 * @param auxiliary 補助のヘッドの教師（補助のヘッドを計算しないなら null）
 * @param tape バックボーンの Attention の自前の演算。逆伝播のために記録するものは学習だけで使い、推論では記録しない
 *     {@link AttentionTape#forwardOnly} を使う。null なら djl-rocm の演算を使う
 * @param probe 段ごとの時間の計測点（学習だけで使う。計測しないなら {@link FitProbe#NONE}）
 */
public record Pass(
    Precision precision,
    boolean[] measured,
    AuxiliaryLabels auxiliary,
    AttentionTape tape,
    FitProbe probe) {

  /**
   * 値の組み合わせを確かめる。
   *
   * @throws IllegalArgumentException 推論に、補助のヘッドの教師、記録する Attention、計測点のどれかを渡したとき
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
   * バックボーンの Attention を自前の演算で行う、推論の順伝播を返す。
   *
   * @param precision 計算の型
   * @param attention 記録しない Attention（{@link AttentionTape#forwardOnly}。null なら djl-rocm の演算を使う）
   * @return 順伝播の種類
   */
  public static Pass inference(Precision precision, AttentionTape attention) {
    return new Pass(precision, null, null, attention, FitProbe.NONE);
  }

  /**
   * 学習の順伝播を返す。
   *
   * @param precision 計算の型
   * @param measured 測った手 C の印（バッチの中の手の順）
   * @param auxiliary 補助のヘッドの教師（null なら補助のヘッドを計算しない）
   * @return 順伝播の種類
   */
  public static Pass training(Precision precision, boolean[] measured, AuxiliaryLabels auxiliary) {
    return training(precision, measured, auxiliary, null);
  }

  /**
   * GPU の学習の順伝播を返す。バックボーンの Attention は自前の HIP の演算で行う。
   *
   * @param precision 計算の型
   * @param measured 測った手 C の印（バッチの中の手の順）
   * @param auxiliary 補助のヘッドの教師（null なら補助のヘッドを計算しない）
   * @param tape Attention の記録（逆伝播は {@link AttentionTape#backward} で行う。null なら djl-rocm の演算を使う）
   * @return 順伝播の種類
   */
  public static Pass training(
      Precision precision, boolean[] measured, AuxiliaryLabels auxiliary, AttentionTape tape) {
    return training(precision, measured, auxiliary, tape, FitProbe.NONE);
  }

  /**
   * 段ごとの時間を測る学習の順伝播を返す（学習器が使う）。
   *
   * @param precision 計算の型
   * @param measured 測った手 C の印（バッチの中の手の順）
   * @param auxiliary 補助のヘッドの教師（null なら補助のヘッドを計算しない）
   * @param tape Attention の記録（null なら djl-rocm の演算を使う）
   * @param probe 段ごとの時間の計測点（計測しないなら {@link FitProbe#NONE}）
   * @return 順伝播の種類
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
