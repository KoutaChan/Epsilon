package com.epsilon.prism3p.variant;

import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.variant.PrismVariant;
import com.epsilon.prism3p.eval.ThreePlayerOpponents;
import com.epsilon.prism3p.input.InputSchema;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 三人打ちのモジュールは ServiceLoader で 1 つだけ見つかり、入力の形式と相手を渡す。classpath にモジュールの無いルールは、名前を添えて拒む。 */
public class PrismVariantTest {

  @Test
  public void theThreePlayerModuleIsFoundOnce() {
    PrismVariant three = PrismVariant.of(Variant.TENHOU_3P);
    Assert.assertTrue(three instanceof ThreePlayer, three.getClass().getName());
    Assert.assertEquals(three.variant(), Variant.TENHOU_3P);
    Assert.assertEquals(List.copyOf(PrismVariant.all()).size(), 1);
    Assert.assertSame(three.input(), InputSchema.FORMAT);
    Assert.assertSame(InputFormat.of(Variant.TENHOU_3P), InputSchema.FORMAT);
    Assert.assertEquals(three.commands(), List.of());
  }

  @Test
  public void aMissingVariantIsRefusedByName() {
    IllegalStateException missing =
        Assert.expectThrows(IllegalStateException.class, () -> PrismVariant.of(Variant.TENHOU_4P));
    Assert.assertEquals(missing.getMessage(), "No PRISM module for TENHOU_4P on the classpath");
  }

  /** 相手はルールベースの打ち手で、CPU で打つ。三人打ちの評価の卓は、ルールベースの打ち手と PRISM のチェックポイントを相手にでき、Nano は拒む。 */
  @Test
  public void opponentsAreTheRuleBotOnCpu() {
    var rules = PrismVariant.of(Variant.TENHOU_3P).opponents();
    Assert.assertSame(rules, ThreePlayerOpponents.RULES);
    Assert.assertEquals(rules.series(), List.of(Series.RULE_BOT));
    Assert.assertEquals(rules.defaultDevice(), "cpu");
    Path absolute = Path.of("models").toAbsolutePath();
    Assert.assertEquals(table(EvaluationTable.HEURISTIC).opponent(), EvaluationTable.HEURISTIC);
    Assert.assertEquals(table("prism:" + absolute).opponent(), "prism:" + absolute);
    SettingsException nano =
        Assert.expectThrows(SettingsException.class, () -> table("nano:" + absolute.resolve("latest")));
    Assert.assertTrue(nano.problems().toString().contains("not NANO"), nano.problems().toString());
  }

  /** 三人打ちの設定で作った評価の卓 t（1 半荘ずつ席を回し、牌山は 6 つ）。 */
  private static TableSpec table(String opponent) {
    String prefix = "evaluation.tables.t.";
    return ThreePlayerSettings.of(
            prefix + "opponent=" + opponent,
            prefix + "seating=BY_GAME",
            prefix + "draws=SEAT_STREAMS",
            prefix + "wallSet=FIXED",
            prefix + "walls=6",
            prefix + "stop=FIXED",
            prefix + "score=RANK",
            prefix + "every=1",
            prefix + "from=1")
        .evaluation()
        .tables()
        .get("t");
  }
}
