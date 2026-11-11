package com.epsilon.prism4p.variant;

import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.variant.PrismVariant;
import com.epsilon.prism4p.eval.FourPlayerOpponents;
import com.epsilon.prism4p.input.InputSchema;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 四人打ちのモジュールは ServiceLoader で 1 つだけ見つかり、入力形式と評価の相手を返す。classpath にモジュールが無いルール（三人打ち）は、
 * ルールの名前を添えて断る。評価の卓の相手には、四人打ちで受け付ける系列（ルールベースの打ち手・Nano・Major）と PRISM を指定できる。
 * チェックポイントを読む相手（Nano・Major・PRISM）は、チェックポイントを絶対パスで指定したときだけ受け付ける。
 */
public class PrismVariantTest {

  @Test
  public void theFourPlayerModuleIsFoundOnce() {
    PrismVariant four = PrismVariant.of(Variant.TENHOU_4P);
    Assert.assertTrue(four instanceof FourPlayer, four.getClass().getName());
    Assert.assertEquals(four.variant(), Variant.TENHOU_4P);
    Assert.assertEquals(List.copyOf(PrismVariant.all()).size(), 1);
    Assert.assertSame(four.input(), InputSchema.FORMAT);
    Assert.assertSame(InputFormat.of(Variant.TENHOU_4P), InputSchema.FORMAT);
    Assert.assertEquals(four.commands(), List.of());
  }

  @Test
  public void aMissingVariantIsRefusedByName() {
    IllegalStateException missing =
        Assert.expectThrows(IllegalStateException.class, () -> PrismVariant.of(Variant.TENHOU_3P));
    Assert.assertEquals(missing.getMessage(), "No PRISM module for TENHOU_3P on the classpath");
  }

  @Test
  public void opponentsArePrismVersionsTheRuleBotNanoAndMajorOnGpu0() {
    var rules = PrismVariant.of(Variant.TENHOU_4P).opponents();
    Assert.assertSame(rules, FourPlayerOpponents.RULES);
    Assert.assertEquals(rules.series(), List.of(Series.RULE_BOT, Series.NANO, Series.MAJOR));
    Assert.assertEquals(rules.defaultDevice(), "gpu0");
    Path absolute = Path.of("models").toAbsolutePath();
    for (String opponent :
        List.of(
            EvaluationTable.HEURISTIC,
            "nano:" + absolute.resolve("latest"),
            "major:" + absolute.resolve("macro614"),
            "prism:" + absolute))
      Assert.assertEquals(
          TestSettings.of(TestSettings.table("x", opponent, 4, 1, 1)).evaluation().tables().get("x").opponent(), opponent);
    for (String opponent : List.of("major:macro614", "prism:models", "nano:"))
      Assert.assertThrows(
          IllegalArgumentException.class, () -> TestSettings.of(TestSettings.table("x", opponent, 4, 1, 1)));
  }
}
