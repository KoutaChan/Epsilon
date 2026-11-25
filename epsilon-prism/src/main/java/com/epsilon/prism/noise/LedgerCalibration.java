package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import java.util.Objects;

/**
 * 帳簿の補正率のセクション。帳簿の補正 β·L に、判断時の公開の局面 o だけから決めた率 h(o) ∈ [0, 1] を掛けて、教師を
 * Ỹ = Y − h(o)·β·L にする。h は {@code source} のコホートの学習の行で当てはめ、今の反復の行を見る前に凍結する。β は今までどおり
 * 半荘の偶奇が逆の行で当てはめる。h が局面だけで決まり、今の行の結果と独立なので、帳簿の条件付き平均が 0 なら Ỹ の期待値は Y と同じ。
 *
 * <p>当てはめ: 自席の順位ごとに、最も近い他家との点差と局の番号の格子（{@code gapGrid} × {@code roundGrid} 点）に係数 θ ∈ [0, 1] を
 * 置き、h(o) を囲む 4 点の双線形補間にする。行 i の、測った手の中で中心化した残差の二乗の平均を J_i(h)、中心化した補正の二乗の平均を A_i
 * として、Σ_i [J_i(h(o_i)) + 2ρ·A_i·h(o_i)] + c·(Σ_i A_i / Nθ)·|θ|² を 0 ≤ θ ≤ 1 で最小にする（ρ = {@code shrink}、
 * c = {@code regularization}、Nθ は係数の数）。
 */
@Choice(key = "mode", doc = "帳簿の補正率を求めるか、教師に使うか")
public sealed interface LedgerCalibration {

  /** 補正率を求めるなら true。 */
  default boolean scales() {
    return this instanceof Scaling;
  }

  /** 教師を補正率で補正するなら true（教師は Ỹ = Y − h(o)·β·L）。 */
  default boolean applies() {
    return this instanceof Apply;
  }

  /** 補正率を求めるモード（SHADOW・APPLY）のキー。 */
  sealed interface Scaling permits Shadow, Apply {
    LedgerCalibrationKind kind();

    LedgerCalibrationSource source();

    int gapGrid();

    int roundGrid();

    double regularization();

    double shrink();
  }

  /** 補正率を求めない。教師は帳簿の補正 Ỹ = Y − β·L のまま。 */
  @Case("OFF")
  record Off() implements LedgerCalibration {}

  /**
   * 補正率を求めて報告するが、教師と持ち越す R² は帳簿の補正 β·L のもののまま。
   *
   * @param kind 補正率の形
   * @param source 補正率を当てはめる行を取るコホート
   * @param gapGrid 最も近い他家との点差の軸に置く格子の点の数
   * @param roundGrid 局の番号の軸に置く格子の点の数
   * @param regularization 係数の二乗の罰の係数 c（罰は c · Σ_i A_i / Nθ · |θ|²。Nθ は係数の数）
   * @param shrink 補正率を 0 へ縮める罰の強さ ρ（行ごとに 2ρ · A_i · h(o_i) を足す）
   */
  @Case("SHADOW")
  record Shadow(
      @Key(doc = "補正率の形", value = "COMMON_SCALE") LedgerCalibrationKind kind,
      @Key(doc = "補正率を当てはめる行を取るコホート", value = "PREVIOUS_COHORT") LedgerCalibrationSource source,
      @Key(doc = "最も近い他家との点差の軸に置く格子の点の数（d / (配給原点 + d) の [0, 1] を等分する）", value = "9")
          @Range(min = 2, max = 17)
          int gapGrid,
      @Key(doc = "局の番号の軸に置く格子の点の数（k / (南場の最後の局の番号 + 1 + k) の [0, 1] を等分する）", value = "3")
          @Range(min = 2, max = 9)
          int roundGrid,
      @Key(doc = "係数の二乗の罰の係数 c（罰は c × 行ごとの補正の分散 A の和 ÷ 係数の数 × 係数の二乗の和）", value = "0.0001")
          @Range(min = 0, minInclusive = false)
          double regularization,
      @Key(doc = "補正率を 0 へ縮める罰の強さ ρ（行ごとに 2ρ × 補正の分散 A × 補正率 h を足す）", value = "0.08") @Range(min = 0)
          double shrink)
      implements LedgerCalibration, Scaling {

    public Shadow {
      Objects.requireNonNull(kind, "kind");
      Objects.requireNonNull(source, "source");
    }
  }

  /**
   * 補正率を求め、教師を Ỹ = Y − h(o)·β·L にする。持ち越す R² は、掛けた補正のホールドアウトの R²。キーは {@link Shadow} と同じ。
   */
  @Case("APPLY")
  record Apply(
      @Key(doc = "補正率の形", value = "COMMON_SCALE") LedgerCalibrationKind kind,
      @Key(doc = "補正率を当てはめる行を取るコホート", value = "PREVIOUS_COHORT") LedgerCalibrationSource source,
      @Key(doc = "最も近い他家との点差の軸に置く格子の点の数（d / (配給原点 + d) の [0, 1] を等分する）", value = "9")
          @Range(min = 2, max = 17)
          int gapGrid,
      @Key(doc = "局の番号の軸に置く格子の点の数（k / (南場の最後の局の番号 + 1 + k) の [0, 1] を等分する）", value = "3")
          @Range(min = 2, max = 9)
          int roundGrid,
      @Key(doc = "係数の二乗の罰の係数 c（罰は c × 行ごとの補正の分散 A の和 ÷ 係数の数 × 係数の二乗の和）", value = "0.0001")
          @Range(min = 0, minInclusive = false)
          double regularization,
      @Key(doc = "補正率を 0 へ縮める罰の強さ ρ（行ごとに 2ρ × 補正の分散 A × 補正率 h を足す）", value = "0.08") @Range(min = 0)
          double shrink)
      implements LedgerCalibration, Scaling {

    public Apply {
      Objects.requireNonNull(kind, "kind");
      Objects.requireNonNull(source, "source");
    }
  }
}
