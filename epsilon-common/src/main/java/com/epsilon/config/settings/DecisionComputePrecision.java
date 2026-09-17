package com.epsilon.config.settings;

/**
 * Decision ネットワークの順伝播計算精度。
 *
 * <p>自動混合精度（autocast）の対象演算に使う精度を指定する。パラメーターや勾配の保存精度は学習方式ごとに決まる。
 * 自己対戦学習ではFLOAT32で保持する。BFLOAT16による事前学習では、演算用パラメーターとその勾配をBFLOAT16、
 * 更新の基準となる重みとオプティマイザー状態をFLOAT32で保持する。固定した推論モデルは指定精度へ変換できる。
 */
public enum DecisionComputePrecision {
  /** 順伝播の浮動小数点演算を32ビット精度で行う。 */
  FLOAT32,

  /** 自動混合精度（autocast）の対象演算に、IEEE 754の半精度浮動小数点形式を使う。 */
  FLOAT16,

  /** 自動混合精度（autocast）の対象演算にbfloat16を使う。 */
  BFLOAT16
}
