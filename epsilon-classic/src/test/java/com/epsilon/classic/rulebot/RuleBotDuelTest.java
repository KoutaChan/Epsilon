package com.epsilon.classic.rulebot;

import com.epsilon.agent.eval.Arena;
import com.epsilon.agent.eval.ArenaPlan;
import com.epsilon.agent.eval.PairScore;
import com.epsilon.agent.eval.Seating;
import com.epsilon.agent.eval.Split;
import com.epsilon.agent.eval.WallSet;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * ルールベースの打ち手どうしの決闘。両方の打ち手をルールベースの打ち手にすると、四人打ち・三人打ちの全ての席の分け方と両方のツモの順で、
 * どの牌山でも両者の差（順位・効用と、席の分け方ごとの値）が丸めなしに 0 になる。
 */
public class RuleBotDuelTest {

  private static final long SEED = 8_088L, BASE = 1L << 40;
  private static final int WALLS = 12;

  private static List<List<Split>> methods(Variant variant) {
    int seats = variant.seats();
    List<Split> both = List.of(Split.oneAgainstRest(variant), new Split(List.of(seats - 1, 1)));
    return seats == 4 ? List.of(both, List.of(new Split(List.of(2, 2)))) : List.of(both);
  }

  @Test(timeOut = 600_000)
  public void theRuleBotAgainstItselfScoresZeroOnEveryWall() {
    try (var table = new TableLoop(new TableSettings(16, 4, 1000L, 4))) {
      for (Variant variant : List.of(Variant.TENHOU_4P, Variant.TENHOU_3P)) {
        for (List<Split> splits : methods(variant)) {
          for (DrawOrder order : DrawOrder.values()) {
            ArenaPlan plan =
                new ArenaPlan(
                    variant,
                    List.of(new RuleBot(variant), new RuleBot(variant)),
                    Seating.EVERY_PLACEMENT,
                    splits,
                    order,
                    WallSet.FIXED,
                    SEED,
                    BASE,
                    WALLS,
                    RankUtility.TENHOU);
            String label = variant + " " + splits + " " + order;
            PairScore score = new PairScore(plan);
            AtomicInteger walls = new AtomicInteger();
            TableLoop.Result result =
                Arena.play(
                    table,
                    plan,
                    (wall, scores, stats) -> {
                      Assert.assertEquals(wall, BASE + walls.getAndIncrement(), label);
                      score.measure(wall, scores);
                      Assert.assertEquals(score.rank(), 0.0, label + " rank of wall " + wall);
                      Assert.assertEquals(score.utility(), 0.0, label + " utility of wall " + wall);
                      for (int split = 0; split < splits.size(); split++) {
                        Assert.assertEquals(score.rank(split), 0.0, label + " split " + split + " of wall " + wall);
                        Assert.assertEquals(score.utility(split), 0.0, label + " split " + split + " of wall " + wall);
                      }
                      return true;
                    },
                    null);
            Assert.assertEquals(walls.get(), WALLS, label);
            Assert.assertEquals(result.games(), (long) WALLS * plan.gamesPerWall(), label);
          }
        }
      }
    }
  }
}
