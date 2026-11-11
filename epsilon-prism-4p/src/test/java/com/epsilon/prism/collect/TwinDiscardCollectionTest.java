package com.epsilon.prism.collect;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 四人打ちの収集で、同値な打牌を切り方のとおりに切る（{@link TwinCollection#assertEveryPath}）。主経路の貪欲・床・探索、写しと分岐の最初の手
 * （候補集合と逸脱）と続きの手の全てで、TSUMOGIRI はツモ牌を、TEDASHI は手牌を切る（立直の打牌も）。手は同値な打牌の組を除けば切り方に
 * 依らず、行も、手の位置を組の手出しに直せば値まで同じ。
 */
public class TwinDiscardCollectionTest {

  private static final List<Long> GAMES = List.of(100L, 101L, 102L, 103L, 104L, 105L, 106L, 107L);

  @Test(timeOut = 900_000)
  public void theFloorPlaysTheChosenTwin() {
    TwinCollection.assertEveryPath(TENHOU_4P, new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE), 21_003L, GAMES);
  }

  @Test(timeOut = 900_000)
  public void theExplorationPlaysTheChosenTwin() {
    TwinCollection.assertEveryPath(TENHOU_4P, new Behaviour(0.05, 0.2, CrnScheme.TWO_STAGE_FLOOR), 21_003L, GAMES);
  }
}
