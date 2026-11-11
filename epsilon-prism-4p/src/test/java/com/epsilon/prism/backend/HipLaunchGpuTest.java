package com.epsilon.prism.backend;

import static com.epsilon.prism.hip.HipKernel.ATTENTION_ALL;
import static com.epsilon.prism.hip.HipKernel.ATTENTION_FIXED;
import static com.epsilon.prism.hip.HipKernel.CANDIDATE;
import static com.epsilon.prism.hip.HipKernel.EMBED;
import static com.epsilon.prism.hip.HipKernel.MLP_ALL;
import static com.epsilon.prism.hip.HipKernel.MLP_FIXED;
import static com.epsilon.prism.hip.HipKernel.MLP_LAST;
import static com.epsilon.prism.hip.HipKernel.RELATION;
import static com.epsilon.prism.hip.HipKernel.REREAD;
import static com.epsilon.prism.hip.HipKernel.REREAD_KEY_VALUE;
import static com.epsilon.prism.hip.HipKernel.VALUE_GRAPH;

import com.epsilon.prism.hip.HipKernel;
import com.epsilon.prism.hip.HipRegion;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * ネットワークの形ごとに、HIP の計画がカーネルを起動する順と回数を確かめる（rocm のグループ。GPU 1 枚）。アームごとに、forward が起動する
 * はずのカーネルと層の列を書き出し、実際の forward と照合する。
 *
 * <ul>
 *   <li>起動するカーネルの種類: forward の計時が NaN でないカーネルが、列に出るカーネルと一致する。
 *   <li>順と回数: 同じ乱数の種で作った別の計画で、列の起動を 1 つずつ {@link com.epsilon.prism.hip.HipPlan#run} で行った s が、forward
 *       の s とビットまで一致する（計画を分けるのは、forward が残した作業領域のせいで、抜けた起動が見えなくならないようにするため）。
 *   <li>同じバッチを 2 回 forward すると、s がビットまで一致する。
 * </ul>
 *
 * <p>v2 の形は、全トークンの層の後に局・席・牌の層（最後の MLP）が 1 つあるだけで、型ごとの層の MLP と、候補層の 2 段目の 2 つのカーネルは
 * 起動しない。イベントのキーとバリューは局・席・牌の層の Attention が n から射影するので、そのための起動もない。v3 の形は層 0 だけが全トークンの
 * 層で、その MLP が層 2 のイベントのキーとバリューを作る（2 段目を持つ形では 2 段目の分も作る）。局・席・牌の Attention を 2 回（層 1 と 2）、
 * 層 1 の MLP を 1 回起動する。2 段目を持つ形では、さらに 2 段目のキーとバリューを作るカーネルと 2 段目のカーネルを起動する。
 */
@Test(groups = "rocm")
public class HipLaunchGpuTest {
  private static final Logger log = LoggerFactory.getLogger(HipLaunchGpuTest.class);

  /**
   * 1 つの起動。
   *
   * @param kernel カーネル
   * @param layer 層（層を持たないカーネルは 0）
   */
  private record Launch(HipKernel kernel, int layer) {}

  public void everyShapeLaunchesItsKernelsInOrder() {
    for (HipFixtures.Arm arm : HipFixtures.Arm.values()) {
      List<Launch> expected = launches(arm);
      long seed = 130 + arm.ordinal();
      float[] replayed;
      try (GpuCase c = HipFixtures.gpuCase(arm, seed)) {
        c.putInput();
        for (Launch launch : expected) {
          c.plan().run(launch.kernel(), launch.layer(), c.hipBatch);
        }
        replayed = c.getFloats(HipRegion.SCORES, c.hipBatch.actions());
      }
      try (GpuCase c = HipFixtures.gpuCase(arm, seed)) {
        c.plan().profile(true);
        float[] first = c.forward();
        float[] millis = c.plan().kernelMillis();
        Set<HipKernel> launched = EnumSet.noneOf(HipKernel.class);
        for (HipKernel kernel : HipKernel.values()) {
          if (!Float.isNaN(millis[kernel.ordinal()])) {
            launched.add(kernel);
          }
        }
        Set<HipKernel> kernels = EnumSet.noneOf(HipKernel.class);
        expected.forEach(launch -> kernels.add(launch.kernel()));
        Assert.assertEquals(launched, kernels, c.name + " kernels");
        Assert.assertEquals(first, replayed, c.name + " s of the launches run one by one");
        Assert.assertEquals(c.forward(), first, c.name + " s of the same batch");
        log.info("PRISM-HIP-LAUNCH {} runs {}", c.name, expected);
      }
    }
  }

  /** アームの forward が行う起動の列。 */
  private static List<Launch> launches(HipFixtures.Arm arm) {
    return switch (arm) {
      case V2_DEPTH_2 ->
          List.of(
              at(EMBED), at(RELATION),
              at(ATTENTION_ALL, 0), at(MLP_ALL, 0),
              at(ATTENTION_FIXED, 1), at(MLP_LAST, 1),
              at(CANDIDATE), at(VALUE_GRAPH));
      case V2_DEPTH_3 ->
          List.of(
              at(EMBED), at(RELATION),
              at(ATTENTION_ALL, 0), at(MLP_ALL, 0), at(ATTENTION_ALL, 1), at(MLP_ALL, 1),
              at(ATTENTION_FIXED, 2), at(MLP_LAST, 2),
              at(CANDIDATE), at(VALUE_GRAPH));
      case V3_ONE_STAGE ->
          List.of(
              at(EMBED), at(RELATION),
              at(ATTENTION_ALL, 0), at(MLP_ALL, 0),
              at(ATTENTION_FIXED, 1), at(MLP_FIXED, 1), at(ATTENTION_FIXED, 2), at(MLP_LAST, 2),
              at(CANDIDATE), at(VALUE_GRAPH));
      case V3, V3_CONTEXT_512 ->
          List.of(
              at(EMBED), at(RELATION),
              at(ATTENTION_ALL, 0), at(MLP_ALL, 0),
              at(ATTENTION_FIXED, 1), at(MLP_FIXED, 1), at(ATTENTION_FIXED, 2), at(MLP_LAST, 2),
              at(REREAD_KEY_VALUE), at(CANDIDATE), at(REREAD), at(VALUE_GRAPH));
    };
  }

  /** 指定した層でのカーネルの起動。 */
  private static Launch at(HipKernel kernel, int layer) {
    return new Launch(kernel, layer);
  }

  /** 層を持たないカーネルの起動。 */
  private static Launch at(HipKernel kernel) {
    return new Launch(kernel, 0);
  }
}
