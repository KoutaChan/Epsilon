package com.epsilon.prism.run;

import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.RowFormat;
import com.google.gson.Gson;
import java.lang.reflect.RecordComponent;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 設定の [epsilon.prism] の鍵は欄の名前と一致し、同梱の既定は検査を通る。知らない鍵と形式は拒み、列は読点で区切る。 */
public class PrismConfigTest {
  private static PrismConfig config(String... pairs) {
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put(PrismConfig.SETTINGS + pairs[i], pairs[i + 1]);
    return PrismConfig.fromSettings(EpsilonSettings.of(values));
  }

  @Test
  public void settingsNameTheConfigFields() {
    Map<String, String> values = EpsilonSettings.defaults().values();
    Set<String> fields = new HashSet<>();
    for (RecordComponent component : PrismConfig.class.getRecordComponents())
      if (!component.getName().equals("format")) {
        fields.add(component.getName());
        Assert.assertTrue(
            values.containsKey(PrismConfig.SETTINGS + component.getName()), component.getName());
      }
    for (String key : values.keySet())
      if (key.startsWith(PrismConfig.SETTINGS))
        Assert.assertTrue(fields.contains(key.substring(PrismConfig.SETTINGS.length())), key);
    PrismConfig defaults = PrismConfig.fromSettings(EpsilonSettings.defaults());
    defaults.validate();
    Assert.assertEquals(defaults.format(), PrismConfig.FORMAT);
    Assert.assertEquals(defaults.seed(), 20261010L);
    Assert.assertEquals(defaults.gamesPerCohort(), 65536);
    Assert.assertEquals(defaults.epsilonB(), 0.005);
    Assert.assertEquals(defaults.holdoutFraction(), 0.05);
    Assert.assertEquals(defaults.maxCandidates(), 6);
    Assert.assertFalse(defaults.measureAllDiscards());
    Assert.assertFalse(defaults.sampling().measuresAll(DecisionClass.DISCARD_MID));
    var all = config("measureAllDiscards", "true").sampling();
    Assert.assertTrue(all.measuresAll(DecisionClass.DISCARD_MID));
    Assert.assertFalse(all.measuresAll(DecisionClass.THREAT));
    // データの記憶: 既定は以前の窓 4 と同じ（4 コホートを全て使う）。keepCohorts は memoryCohorts 以上
    Assert.assertEquals(defaults.memoryCohorts(), 4);
    Assert.assertEquals(defaults.memoryKeep(0), 1.0);
    Assert.assertEquals(defaults.memoryKeep(3), 1.0);
    Assert.assertEquals(config("memoryDecay", "0.5").memoryKeep(2), 0.25);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> config("memoryCohorts", "8").validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("memoryDecay", "0").validate());
    // 方策の記憶: 既定は平均しない。名前の間違いは拒む
    Assert.assertFalse(defaults.policyMemory().enabled());
    Assert.assertEquals(defaults.collectWith(), Averaging.AVERAGE);
    Assert.assertEquals(defaults.learnFrom(), Averaging.ITERATE);
    Assert.assertTrue(config("averaging", "ema").policyMemory().enabled());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("averaging", "mean").validate());
    Assert.assertThrows(
        IllegalArgumentException.class, () -> config("collectWith", "latest").validate());
    Assert.assertEquals(defaults.fitSteps(), 2000);
    Assert.assertEquals(defaults.distillSteps(), 1000);
    Assert.assertEquals(defaults.fitEpochs(), 0.6);
    Assert.assertEquals(defaults.distillEpochs(), 1.0);
    Assert.assertEquals(defaults.lrPeak(), 2e-4);
    Assert.assertEquals(defaults.kappa(), 0.0);
    Assert.assertEquals(defaults.klReference(), PrismConfig.KL_UNIFORM);
    Assert.assertTrue(defaults.fitSettings(1).uniformReference());
    Assert.assertFalse(config("klReference", "prior").fitSettings(1).uniformReference());
    Assert.assertThrows(
        IllegalArgumentException.class, () -> config("klReference", "flat").validate());
    // 学習器: 既定は交差適合（生徒 A・B と蒸留）。single は全データの生徒 1 本。名前の間違いは拒む
    Assert.assertEquals(defaults.learner(), PrismConfig.LEARNER_CROSSFIT);
    Assert.assertFalse(defaults.singleStudent());
    Assert.assertTrue(config("learner", "single").singleStudent());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("learner", "solo").validate());
    // 歩数は上限と周回数の小さい方（行がないか上限 0 なら 0、それ以外は 1 以上）
    Assert.assertEquals(PrismConfig.steps(2000, 0.6, 1_000_000, 4096), 147);
    Assert.assertEquals(PrismConfig.steps(100, 0.6, 1_000_000, 4096), 100);
    Assert.assertEquals(PrismConfig.steps(2000, 0.6, 10, 4096), 1);
    Assert.assertEquals(PrismConfig.steps(2000, 0.6, 0, 4096), 0);
    Assert.assertEquals(PrismConfig.steps(0, 0.6, 1_000_000, 4096), 0);
    Assert.assertTrue(defaults.certify());
    Assert.assertEquals(defaults.certifyTolerance(), 0.0005);
    Assert.assertTrue(defaults.startSnapshot().isEmpty());
    // config.json（Gson）を通っても同じ設定に戻る
    Assert.assertEquals(new Gson().fromJson(new Gson().toJson(defaults), PrismConfig.class), defaults);
    Assert.assertEquals(
        config("collectDevices", "gpu0, gpu1").collectDevices(), List.of("gpu0", "gpu1"));
    for (String key : new String[] {"games", "format"})
      Assert.assertThrows(IllegalArgumentException.class, () -> config(key, "1"));
    Assert.assertThrows(IllegalArgumentException.class, () -> config("hidden", "wide"));
    Assert.assertFalse(config("certify", "false").certify());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("certify", "yes"));
    Assert.assertThrows(IllegalArgumentException.class, () -> config("keepCohorts", "2").validate());
    // Φ: 既定は GRP で、train は grpCheckpoint がなければ拒む（点数の順位の代用は明示したときだけ）
    Assert.assertEquals(defaults.phi(), PrismConfig.PHI_GRP);
    Assert.assertEquals(defaults.certifyMaxOutsideShare(), 0.02);
    Assert.assertThrows(IllegalArgumentException.class, defaults::validateTraining);
    config("phi", PrismConfig.PHI_SCORE_RANK).validateTraining();
    config("grpCheckpoint", "/grp").validateTraining();
    Assert.assertThrows(IllegalArgumentException.class, () -> config("phi", "gpr").validate());
  }

  /** 半荘 ID とコホートの種の式（同じ種なら同じ牌山）。 */
  @Test
  public void cohortGamesFollowTheSeed() {
    Assert.assertEquals(
        PrismConfig.cohortGames(2, 3),
        List.of(
            (long) PrismConfig.MAX_GAMES_PER_COHORT,
            PrismConfig.MAX_GAMES_PER_COHORT + 1L,
            PrismConfig.MAX_GAMES_PER_COHORT + 2L));
    Assert.assertNotEquals(PrismConfig.cohortSeed(1, 1), PrismConfig.cohortSeed(1, 2));
  }

  /**
   * 輪が行を選ぶ条件: 生徒は自分の組の TRAIN のラベル行で |C| ≥ 2、全データの生徒は両方の組の TRAIN のラベル行で |C| ≥ 2、蒸留は TRAIN の
   * ラベル行、認証は根、保留は HOLDOUT のラベル行で |C| ≥ 2。
   */
  @Test
  public void pipelineSelectsTheRightRows() {
    var label =
        new RowFormat.Index(
            40,
            PrismRow.Purpose.TRAIN,
            PrismRow.Group.A,
            PrismRow.Source.ROLLIN,
            com.epsilon.prism.core.DecisionClass.CALL,
            2);
    Assert.assertTrue(Pipeline.studentRows(PrismRow.Group.A).test(label));
    Assert.assertFalse(Pipeline.studentRows(PrismRow.Group.B).test(label));
    Assert.assertTrue(Pipeline.distillRows().test(label));
    Assert.assertTrue(Pipeline.trainRows().test(label));
    Assert.assertFalse(Pipeline.rootRows().test(label));
    Assert.assertFalse(Pipeline.holdoutRows().test(label));
    var single =
        new RowFormat.Index(
            40,
            PrismRow.Purpose.TRAIN,
            PrismRow.Group.A,
            PrismRow.Source.MAIN,
            com.epsilon.prism.core.DecisionClass.DISCARD_MID,
            1);
    Assert.assertFalse(Pipeline.studentRows(PrismRow.Group.A).test(single));
    Assert.assertFalse(Pipeline.trainRows().test(single));
    Assert.assertTrue(Pipeline.distillRows().test(single));
    var root =
        new RowFormat.Index(
            40,
            PrismRow.Purpose.HOLDOUT,
            PrismRow.Group.B,
            PrismRow.Source.ROOT,
            com.epsilon.prism.core.DecisionClass.RIICHI,
            2);
    Assert.assertTrue(Pipeline.rootRows().test(root));
    Assert.assertFalse(Pipeline.distillRows().test(root));
    Assert.assertFalse(Pipeline.trainRows().test(root));
    Assert.assertFalse(Pipeline.holdoutRows().test(root));
    var holdout =
        new RowFormat.Index(
            40,
            PrismRow.Purpose.HOLDOUT,
            PrismRow.Group.B,
            PrismRow.Source.MAIN,
            com.epsilon.prism.core.DecisionClass.THREAT,
            3);
    Assert.assertTrue(Pipeline.holdoutRows().test(holdout));
    Assert.assertFalse(Pipeline.studentRows(PrismRow.Group.B).test(holdout));
    Assert.assertFalse(Pipeline.trainRows().test(holdout));
  }
}
