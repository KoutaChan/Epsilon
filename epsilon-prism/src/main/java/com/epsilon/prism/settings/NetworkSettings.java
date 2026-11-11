package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;

/** ゼロから学習するバージョン 1 のネットワークの形。後のバージョンは前のバージョンの形を引き継ぐ。 */
@Choice(key = "shape", doc = "ネットワークの形")
public sealed interface NetworkSettings {

  /** ネットワークの形。 */
  PrismShape shape();

  /**
   * v2 の構造（H 256・F 512・G 256）。
   *
   * @param depth バックボーンの層の数
   */
  @Case("V2")
  record V2(
      @Key(doc = "バックボーンの層の数", value = "2") @Range(min = PrismShape.MIN_DEPTH, max = PrismShape.MAX_DEPTH) int depth)
      implements NetworkSettings {

    @Override
    public PrismShape shape() {
      return ShapeArm.V2.shape(depth);
    }
  }

  /** トークンの型ごとに FFN を持つ v3 の構造（深さ 3）。 */
  @Case("V3")
  record V3() implements NetworkSettings {

    @Override
    public PrismShape shape() {
      return ShapeArm.V3.shape(ShapeArm.V3_DEPTH);
    }
  }

  /** v3 の構造から、候補層の 2 段目を外した形（深さ 3）。 */
  @Case("V3_ONE_STAGE")
  record V3OneStage() implements NetworkSettings {

    @Override
    public PrismShape shape() {
      return ShapeArm.V3_ONE_STAGE.shape(ShapeArm.V3_DEPTH);
    }
  }

  /** v3 の構造で、局と席の FFN の幅を 512 にした形（深さ 3）。 */
  @Case("V3_CONTEXT_512")
  record V3Context512() implements NetworkSettings {

    @Override
    public PrismShape shape() {
      return ShapeArm.V3_CONTEXT_512.shape(ShapeArm.V3_DEPTH);
    }
  }
}
