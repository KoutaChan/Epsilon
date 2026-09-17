package com.epsilon.config.settings;

/** 行動選択モデルの推論を、通常のDJL演算で実行するか、djl-rocmの融合演算で実行するかを指定する。 */
public enum DecisionInferenceFusionMode {
  /** 通常のDJL演算を組み合わせて実行する基準経路。 */
  EAGER,

  /** 凍結済みGPU推論向けの永続Fusion 実行計画で実行する経路。 */
  FUSION
}
