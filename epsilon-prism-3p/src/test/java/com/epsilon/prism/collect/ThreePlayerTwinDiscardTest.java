package com.epsilon.prism.collect;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.core.TwinDiscard;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの収集で、同値な打牌を切り方のとおりに切る（{@link TwinCollection#assertEveryPath}）。確かめることは四人打ちと同じ。三人打ちの設定の
 * 既定も、ツモ切り。
 */
public class ThreePlayerTwinDiscardTest {

  private static final List<Long> GAMES = List.of(200L, 201L, 202L, 203L, 204L, 205L, 206L, 207L);

  @Test
  public void theThreePlayerDefaultIsTsumogiri() {
    Assert.assertEquals(ThreePlayerSettings.of().behaviour().twinDiscard(), TwinDiscard.TSUMOGIRI);
  }

  @Test(timeOut = 900_000)
  public void theFloorPlaysTheChosenTwin() {
    TwinCollection.assertEveryPath(TENHOU_3P, new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE), 31_007L, GAMES);
  }

  @Test(timeOut = 900_000)
  public void theExplorationPlaysTheChosenTwin() {
    TwinCollection.assertEveryPath(TENHOU_3P, new Behaviour(0.05, 0.2, CrnScheme.TWO_STAGE_FLOOR), 31_007L, GAMES);
  }
}
