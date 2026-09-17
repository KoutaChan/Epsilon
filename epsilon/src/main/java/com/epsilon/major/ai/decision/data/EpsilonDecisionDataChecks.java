package com.epsilon.major.ai.decision.data;

/** Decision/GRP の学習データで共有する不正な値を見つけた時点で例外を送出する検証。 */
public final class EpsilonDecisionDataChecks {

  private EpsilonDecisionDataChecks() {}

  /**
   * 単一の float が有限値であることを要求し、その値をそのまま返す。
   *
   * <p>NaN/Infinity は学習データの破損として扱う。呼び出し側で 0 に丸めると、どの入力が壊れたかを失うため、 データ検証例外として即時に落とす。
   *
   * @param value 検証する値
   * @param label 例外へ含めるフィールド名
   * @return 検証済みの元値
   */
  public static float requireFinite(float value, String label) {
    if (!Float.isFinite(value)) {
      throw new EpsilonDecisionDataException(label + " contains non-finite value");
    }
    return value;
  }

  /**
   * 配列内の全要素が有限値であることを要求し、元の配列を返す。
   *
   * @param values 検証する配列
   * @param label 例外へ含めるフィールド名
   * @return 検証済みの同一配列
   */
  public static float[] requireFinite(float[] values, String label) {
    for (float value : values) {
      requireFinite(value, label);
    }
    return values;
  }

  /**
   * 非負かつ有限値の float を要求し、その値を返す。
   *
   * <p>確率、エントロピー、サンプル件数由来の比率など、負値が特別な意味を持つ値ではなく入力破損を意味する値に使う。
   *
   * @param value 検証する値
   * @param label 例外へ含めるフィールド名
   * @return 検証済みの元値
   */
  public static float requireNonNegativeFinite(float value, String label) {
    requireFinite(value, label);
    if (value < 0.0f) {
      throw new EpsilonDecisionDataException(label + " must be non-negative: " + value);
    }
    return value;
  }

  /**
   * 配列の指定インデックスを範囲検証してから有限値として返す。
   *
   * <p>方策配列の位置、効用の定義、行動ごとの方策などの直接参照で、範囲外や有限でない値を 0 に丸めないための補助処理。
   *
   * @param values 参照する配列
   * @param index 読み取るインデックス
   * @param label 例外へ含めるフィールド名
   * @return 指定位置の検証済み値
   */
  public static float requireFiniteAt(float[] values, int index, String label) {
    if (index < 0 || index >= values.length) {
      throw new IllegalArgumentException(label + " index out of range: " + index);
    }
    return requireFinite(values[index], label);
  }

  /**
   * 配列長が固定値と一致し、全要素が有限値であることを要求して元の配列を返す。
   *
   * @param values 検証する配列
   * @param expectedLength 必須要素数
   * @param label 例外へ含めるフィールド名
   * @return 検証済みの同一配列
   */
  public static float[] requireFixedLengthFinite(float[] values, int expectedLength, String label) {
    if (values.length != expectedLength) {
      throw new IllegalArgumentException(
          label + " length must be " + expectedLength + ": " + values.length);
    }
    return requireFinite(values, label);
  }
}
