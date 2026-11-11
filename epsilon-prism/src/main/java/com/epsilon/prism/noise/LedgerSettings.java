package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import java.util.Set;

/**
 * 運の精算の帳簿のセクション。行に帳簿を書くか、係数を当てはめるか、教師を補正するかを、1 つのキー（{@code mode}）で選ぶ。モードの
 * キーと帳簿を計算するキーは収集から効き、係数を当てはめるキーは学習から効く。
 */
@Choice(key = "mode", doc = "帳簿を書くか、監査するか、教師の補正に使うか")
public sealed interface LedgerSettings {

  /** 行に帳簿を書くなら true。 */
  default boolean records() {
    return this instanceof Recording;
  }

  /** 帳簿の係数を当てはめて報告するなら true。 */
  default boolean fits() {
    return this instanceof Fitting;
  }

  /** 教師を帳簿で補正するなら true（教師は Ỹ = Y − β·L）。 */
  default boolean corrects() {
    return this instanceof Apply;
  }

  /** 帳簿を行に書くモード（RECORD・AUDIT・APPLY）で、収集から効くキー。 */
  sealed interface Recording permits RecordOnly, Audit, Apply {
    Set<LedgerGroup> compute();

    double pointClip();

    double policyClip();

    int slopeStep();

    SlopeRefresh slopeRefresh();
  }

  /** 係数を当てはめるモード（AUDIT・APPLY）で、学習から効くキー。 */
  sealed interface Fitting permits Audit, Apply {
    Set<LedgerGroup> apply();

    LedgerCells cells();

    double ridge();

    int minRowsPerColumn();

    CoefficientSource coefficients();

    double biasZ();
  }

  /** 帳簿を書かない。教師は生の Y。 */
  @Case("OFF")
  record Off() implements LedgerSettings {}

  /**
   * 帳簿を行に書くが、係数は当てはめない。教師は生の Y。
   *
   * @param compute 帳簿で計算する組（空にしない）
   * @param pointClip 点の列を切る幅（千点）
   * @param policyClip 方策の列で手の値を切る幅（効用）
   * @param slopeStep 傾きの列で点をずらす幅（100 点の倍数）
   * @param slopeRefresh 傾きの列を作り直す時点
   */
  @Case("RECORD")
  record RecordOnly(
      @Key(doc = "帳簿で計算する組（空にしない）", value = "[IMPROVE, POINTS, DORA, URA, POLICY]") Set<LedgerGroup> compute,
      @Key(doc = "点の列を切る幅（千点）", value = "32") @Range(min = 0, minInclusive = false) double pointClip,
      @Key(doc = "方策の列で手の値を切る幅（効用）", value = "8") @Range(min = 0, minInclusive = false) double policyClip,
      @Key(doc = "傾きの列で点をずらす幅（100 点の倍数）", value = "1000") @Range(min = 100, max = 8000) int slopeStep,
      @Key(doc = "傾きの列を作り直す時点", value = "DEPOSIT") SlopeRefresh slopeRefresh)
      implements LedgerSettings, Recording {

    public RecordOnly {
      compute = Set.copyOf(compute);
      requireCollection(compute, slopeStep);
    }
  }

  /**
   * 帳簿を行に書き、係数を当てはめて報告する。教師は生の Y。
   *
   * @param apply 教師の補正に使う組（計算する組の部分集合で、空にしない）
   * @param cells 係数を当てはめる区分
   * @param ridge 係数を縮める強さ（リッジの強さ）
   * @param minRowsPerColumn 区分に要る行の数（当てはめる列 1 つあたり）
   * @param coefficients 行に掛ける係数をどの行で当てはめるか
   * @param biasZ 偏りを警告するしきい値（標準誤差の倍数）
   */
  @Case("AUDIT")
  record Audit(
      @Key(doc = "帳簿で計算する組（空にしない）", value = "[IMPROVE, POINTS, DORA, URA, POLICY]") Set<LedgerGroup> compute,
      @Key(doc = "点の列を切る幅（千点）", value = "32") @Range(min = 0, minInclusive = false) double pointClip,
      @Key(doc = "方策の列で手の値を切る幅（効用）", value = "8") @Range(min = 0, minInclusive = false) double policyClip,
      @Key(doc = "傾きの列で点をずらす幅（100 点の倍数）", value = "1000") @Range(min = 100, max = 8000) int slopeStep,
      @Key(doc = "傾きの列を作り直す時点", value = "DEPOSIT") SlopeRefresh slopeRefresh,
      @Key(doc = "教師の補正に使う組（計算する組の部分集合で、空にしない）", value = "[IMPROVE, POINTS, DORA, URA, POLICY]", change = Change.FIT)
          Set<LedgerGroup> apply,
      @Key(doc = "係数を当てはめる区分", value = "CLASS", change = Change.FIT) LedgerCells cells,
      @Key(doc = "係数を縮める強さ（リッジの強さ）", value = "0.001", change = Change.FIT) @Range(min = 0, minInclusive = false)
          double ridge,
      @Key(doc = "区分に要る行の数（当てはめる列 1 つあたり）", value = "32", change = Change.FIT) @Range(min = 1)
          int minRowsPerColumn,
      @Key(doc = "行に掛ける係数を、どの行で当てはめるか", value = "OTHER_PARITY", change = Change.FIT) CoefficientSource coefficients,
      @Key(doc = "偏りを警告するしきい値（標準誤差の倍数）", value = "3.0", change = Change.FIT) @Range(min = 0, minInclusive = false)
          double biasZ)
      implements LedgerSettings, Recording, Fitting {

    public Audit {
      compute = Set.copyOf(compute);
      apply = Set.copyOf(apply);
      requireCollection(compute, slopeStep);
      requireFit(compute, apply);
    }
  }

  /** 帳簿を行に書き、係数を当てはめて、教師を Ỹ = Y − β·L に補正する。キーは {@link Audit} と同じ。 */
  @Case("APPLY")
  record Apply(
      @Key(doc = "帳簿で計算する組（空にしない）", value = "[IMPROVE, POINTS, DORA, URA, POLICY]") Set<LedgerGroup> compute,
      @Key(doc = "点の列を切る幅（千点）", value = "32") @Range(min = 0, minInclusive = false) double pointClip,
      @Key(doc = "方策の列で手の値を切る幅（効用）", value = "8") @Range(min = 0, minInclusive = false) double policyClip,
      @Key(doc = "傾きの列で点をずらす幅（100 点の倍数）", value = "1000") @Range(min = 100, max = 8000) int slopeStep,
      @Key(doc = "傾きの列を作り直す時点", value = "DEPOSIT") SlopeRefresh slopeRefresh,
      @Key(doc = "教師の補正に使う組（計算する組の部分集合で、空にしない）", value = "[IMPROVE, POINTS, DORA, URA, POLICY]", change = Change.FIT)
          Set<LedgerGroup> apply,
      @Key(doc = "係数を当てはめる区分", value = "CLASS", change = Change.FIT) LedgerCells cells,
      @Key(doc = "係数を縮める強さ（リッジの強さ）", value = "0.001", change = Change.FIT) @Range(min = 0, minInclusive = false)
          double ridge,
      @Key(doc = "区分に要る行の数（当てはめる列 1 つあたり）", value = "32", change = Change.FIT) @Range(min = 1)
          int minRowsPerColumn,
      @Key(doc = "行に掛ける係数を、どの行で当てはめるか", value = "OTHER_PARITY", change = Change.FIT) CoefficientSource coefficients,
      @Key(doc = "偏りを警告するしきい値（標準誤差の倍数）", value = "3.0", change = Change.FIT) @Range(min = 0, minInclusive = false)
          double biasZ)
      implements LedgerSettings, Recording, Fitting {

    public Apply {
      compute = Set.copyOf(compute);
      apply = Set.copyOf(apply);
      requireCollection(compute, slopeStep);
      requireFit(compute, apply);
    }
  }

  private static void requireCollection(Set<LedgerGroup> compute, int slopeStep) {
    if (compute.isEmpty()) {
      throw new InvalidSetting("compute", "a ledger that records computes at least one group");
    }
    if (slopeStep % 100 != 0) {
      throw new InvalidSetting("slopeStep", "slopeStep must be a multiple of 100: " + slopeStep);
    }
  }

  private static void requireFit(Set<LedgerGroup> compute, Set<LedgerGroup> apply) {
    if (apply.isEmpty() || !compute.containsAll(apply)) {
      throw new InvalidSetting("apply", "apply " + apply + " must be a non-empty subset of compute " + compute);
    }
  }
}
