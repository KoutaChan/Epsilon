package com.epsilon.classic;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.rulebot.RuleBot;
import com.epsilon.classic.rulebot.RuleBotFamily;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/**
 * ServiceLoader が classic の 2 つの系列とルールベースの打ち手を見つけること、開くメソッドが重みを読む前にチェックポイントのヘッダーを
 * 照合することを確かめる。ヘッダーのテストは model.json だけを置いたディレクトリを使い、ネットワークは読まない。
 */
public class ModelFamiliesTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final PolicyOptions CPU = PolicyOptions.defaults();

  @Test
  public void serviceLoaderFindsEveryClassicSeries() {
    assertTrue(ModelFamily.of(Series.NANO) instanceof ClassicFamily.Nano);
    assertTrue(ModelFamily.of(Series.MAJOR) instanceof ClassicFamily.Major);
    assertTrue(ModelFamily.of(Series.RULE_BOT) instanceof RuleBotFamily);
    assertEquals(ModelFamily.of(Series.NANO).series(), Series.NANO);
    assertEquals(ModelFamily.of(Series.MAJOR).series(), Series.MAJOR);
  }

  @Test
  public void theRuleBotOpensWithoutACheckpoint() throws IOException {
    ModelFamily family = ModelFamily.of(Series.RULE_BOT);
    try (Policy policy = family.open(null, VARIANT, CPU)) {
      assertTrue(policy instanceof RuleBot);
      assertTrue(policy instanceof InlinePolicy);
      assertEquals(policy.label(), "heuristic");
      assertEquals(policy.variant(), VARIANT);
    }
    try (NetworkRuntime runtime = new NetworkRuntime(1, 1, 1);
        Policy policy = family.open(null, VARIANT, CPU, runtime)) {
      assertTrue(policy instanceof RuleBot);
    }
    expectThrows(IllegalArgumentException.class, () -> family.open(Path.of("nano-1"), VARIANT, CPU));
  }

  @Test
  public void openingRefusesAnotherVariantFingerprintOrArchitecture() throws IOException {
    Path root = Files.createTempDirectory("classic-headers");
    try {
      String fingerprint = ClassicSchema.of(Architecture.NANO).fingerprint();
      Path nano = header(root.resolve("nano"), Architecture.NANO, VARIANT, fingerprint);
      Path stale = header(root.resolve("stale"), Architecture.NANO, VARIANT, "0123456789abcdef");
      ModelFamily family = ModelFamily.of(Series.NANO);
      IllegalStateException variant =
          expectThrows(IllegalStateException.class, () -> family.open(nano, Variant.TENHOU_3P, CPU));
      assertTrue(variant.getMessage().contains("TENHOU_3P"), variant.getMessage());
      IllegalStateException inputs = expectThrows(IllegalStateException.class, () -> family.open(stale, VARIANT, CPU));
      assertTrue(inputs.getMessage().contains("fingerprint"), inputs.getMessage());
      IllegalStateException architecture =
          expectThrows(IllegalStateException.class, () -> ModelFamily.of(Series.MAJOR).open(nano, VARIANT, CPU));
      assertTrue(architecture.getMessage().contains("NANO"), architecture.getMessage());
      expectThrows(IllegalArgumentException.class, () -> family.open(null, VARIANT, CPU));
    } finally {
      try (Stream<Path> files = Files.walk(root)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
          Files.delete(file);
        }
      }
    }
  }

  /** model.json だけを置いたチェックポイントのディレクトリを作る。 */
  private static Path header(Path directory, Architecture architecture, Variant variant, String fingerprint)
      throws IOException {
    Files.createDirectories(directory);
    Files.writeString(
        directory.resolve("model.json"),
        "{\"architecture\": \"" + architecture + "\", \"variant\": \"" + variant + "\", \"utility\": \""
            + RankUtility.TENHOU + "\", \"inputFingerprint\": \"" + fingerprint + "\", \"step\": 0}");
    return directory;
  }
}
