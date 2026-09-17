package com.epsilon.major.ai.policy;

import com.epsilon.major.config.settings.DecisionSettings;

/** Decision・GRP・Belief の学習で共有する、学習率・重み減衰・勾配上限の定数。 */
public final class EpsilonTrainingConstants {

  private EpsilonTrainingConstants() {}

  /** Decision オプティマイザーの基準学習率。 */
  public static final float LEARNING_RATE = DecisionSettings.defaults().learningRate();

  /** Decision オプティマイザーの勾配更新から分離した重み減衰。 */
  public static final float WEIGHT_DECAY = DecisionSettings.defaults().weightDecay();

  /** Decision オプティマイザーの全勾配クリップ。 */
  public static final float GRAD_CLIP = DecisionSettings.defaults().gradClip();
}
