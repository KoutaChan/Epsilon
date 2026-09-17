package com.epsilon.nano.ai.policy;

import com.epsilon.nano.config.settings.DecisionSettings;

/** Decision、GRP、Belief の学習で共有するハイパーパラメーター。 */
public final class EpsilonTrainingConstants {

  private EpsilonTrainingConstants() {}

  /** Decision オプティマイザーの基準学習率。 */
  public static final float LEARNING_RATE = DecisionSettings.defaults().learningRate();

  /** Decision オプティマイザーの 勾配更新と分離した重み減衰。 */
  public static final float WEIGHT_DECAY = DecisionSettings.defaults().weightDecay();

  /** Decision オプティマイザーの全勾配クリップ。 */
  public static final float GRAD_CLIP = DecisionSettings.defaults().gradClip();
}
