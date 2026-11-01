package com.epsilon.prism.run;

import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.InferencePath;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 設定の [epsilon.prism] の鍵は欄の名前と一致し、同梱の既定は本番の系譜の値で検査を通る。知らない鍵と形式は設定でも config.json でも拒み、
 * config.json は欄とちょうど同じ鍵を持つ。列は読点で区切る。
 */
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
    Assert.assertEquals(defaults.iterations(), 1000);
    Assert.assertEquals(defaults.inferBatch(), 2048);
    Assert.assertEquals(defaults.epsilonB(), 0.005);
    Assert.assertEquals(defaults.holdoutFraction(), 0.05);
    Assert.assertEquals(defaults.labelRateCommit(), 0.104);
    // オーラスの和了は確約の類と別の率で、既定は全てラベル行
    Assert.assertEquals(defaults.labelRateFinalEnd(), 1.0);
    Assert.assertEquals(defaults.sampling().labelRate(DecisionClass.END), 0.104);
    Assert.assertEquals(defaults.sampling().labelRate(DecisionClass.FINAL_END), 1.0);
    Assert.assertEquals(
        config("labelRateFinalEnd", "0.5").sampling().labelRate(DecisionClass.FINAL_END), 0.5);
    Assert.assertEquals(defaults.maxCandidates(), 6);
    Assert.assertEquals(defaults.keepCohorts(), 16);
    Assert.assertEquals(defaults.fitSteps(), 2000);
    Assert.assertEquals(defaults.fitEpochs(), 1.0);
    Assert.assertEquals(defaults.lrPeak(), 0.0012);
    Assert.assertEquals(defaults.lrFloor(), 6e-5);
    Assert.assertEquals(defaults.filterGain(), 0.9);
    Assert.assertEquals(defaults.filterTau(), 4.0);
    Assert.assertEquals(defaults.filterPower(), 0.6666667);
    Assert.assertEquals(defaults.filterLrMode(), FilterLrMode.GAIN);
    Assert.assertEquals(defaults.evalGames(), 8192);
    // 方策の記憶: 既定は平均しない（収集も生徒も学習の網）。名前の間違いは拒む
    Assert.assertFalse(defaults.policyMemory().enabled());
    Assert.assertEquals(defaults.averageFrom(), 16);
    Assert.assertEquals(defaults.collectWith(), Averaging.ITERATE);
    Assert.assertEquals(defaults.learnFrom(), Averaging.ITERATE);
    Assert.assertTrue(config("averaging", "ema").policyMemory().enabled());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("averaging", "mean").validate());
    Assert.assertThrows(
        IllegalArgumentException.class, () -> config("collectWith", "latest").validate());
    // 歩数は上限と周回数の小さい方（行がないか上限 0 なら 0、それ以外は 1 以上）
    Assert.assertEquals(PrismConfig.steps(2000, 0.6, 1_000_000, 4096), 147);
    Assert.assertEquals(PrismConfig.steps(100, 0.6, 1_000_000, 4096), 100);
    Assert.assertEquals(PrismConfig.steps(2000, 0.6, 10, 4096), 1);
    Assert.assertEquals(PrismConfig.steps(2000, 0.6, 0, 4096), 0);
    Assert.assertEquals(PrismConfig.steps(0, 0.6, 1_000_000, 4096), 0);
    // config.json（Gson）を通っても同じ設定に戻る
    Assert.assertEquals(PrismConfig.read(new Gson().toJson(defaults)), defaults);
    Assert.assertEquals(
        config("collectDevices", "gpu0, gpu1").collectDevices(), List.of("gpu0", "gpu1"));
    Assert.assertThrows(IllegalArgumentException.class, () -> config("depth", "two"));
    Assert.assertThrows(IllegalArgumentException.class, () -> config("frontier", "yes"));
    Assert.assertThrows(IllegalArgumentException.class, () -> config("keepCohorts", "0").validate());
    // Φ: 既定は GRP で、train は grpCheckpoint がなければ拒む（点数の順位の代用は明示したときだけ）
    Assert.assertEquals(defaults.phi(), PrismConfig.PHI_GRP);
    Assert.assertThrows(IllegalArgumentException.class, defaults::validateTraining);
    config("phi", PrismConfig.PHI_SCORE_RANK).validateTraining();
    config("grpCheckpoint", "/grp").validateTraining();
    Assert.assertThrows(IllegalArgumentException.class, () -> config("phi", "gpr").validate());
  }

  /**
   * epsilon.prism. で始まる知らない鍵（打ち間違い・なくなった設定・点で区切った名前・形式）は拒む。HIP の検査つきの組み立てを選ぶ系の性質だけは
   * 設定の鍵でなくても通す。
   */
  @Test
  public void unknownSettingsAreRejected() {
    for (String key :
        new String[] {"games", "format", "trainRoots", "eventLedger", "v2Inference", "v2.balancedLanes"}) {
      IllegalArgumentException failure =
          Assert.expectThrows(IllegalArgumentException.class, () -> config(key, "1"));
      Assert.assertTrue(failure.getMessage().contains(PrismConfig.SETTINGS + key), failure.getMessage());
    }
    Map<String, String> checked = Map.of(HipLibrary.CHECKED_PROPERTY, "true");
    PrismConfig.requireKnownSettings(EpsilonSettings.of(checked));
    PrismConfig.fromSettings(EpsilonSettings.of(checked)).validate();
  }

  /** config.json は欄とちょうど同じ鍵を持つ: 知らない鍵・欠けた鍵・null は拒み、形式の違いは検査で拒む。 */
  @Test
  public void configJsonMatchesTheFieldsExactly() {
    Gson gson = new GsonBuilder().serializeNulls().create();
    JsonObject tree = gson.toJsonTree(PrismConfig.fromSettings(EpsilonSettings.defaults())).getAsJsonObject();
    JsonObject unknown = tree.deepCopy();
    unknown.addProperty("trainRoots", true);
    Assert.assertThrows(IllegalArgumentException.class, () -> PrismConfig.read(gson.toJson(unknown)));
    JsonObject missing = tree.deepCopy();
    missing.remove("filterLrMode");
    Assert.assertThrows(IllegalArgumentException.class, () -> PrismConfig.read(gson.toJson(missing)));
    JsonObject empty = tree.deepCopy();
    empty.add("inference", JsonNull.INSTANCE);
    Assert.assertThrows(IllegalArgumentException.class, () -> PrismConfig.read(gson.toJson(empty)));
    JsonObject older = tree.deepCopy();
    older.addProperty("format", PrismConfig.FORMAT - 1);
    PrismConfig read = PrismConfig.read(gson.toJson(older));
    Assert.assertThrows(IllegalArgumentException.class, read::validate);
    Assert.assertThrows(IllegalArgumentException.class, () -> PrismConfig.read("[]"));
  }

  /** 収集は打牌の類で全ての合法手を測り（確約の類は候補集合）、ラベルの枠ごとに運の精算の帳簿を書く。 */
  @Test
  public void samplingMeasuresAllDiscardsWithTheLedger() {
    var sampling = PrismConfig.fromSettings(EpsilonSettings.defaults()).sampling();
    for (DecisionClass c : DecisionClass.values())
      Assert.assertEquals(sampling.measuresAll(c), !c.commitment(), c.name());
    Assert.assertTrue(sampling.eventLedger());
  }

  /** 網の推論の道: 既定は HIP、EAGER を選べ、名前の間違いは拒む。 */
  @Test
  public void inferenceDefaultsToHip() {
    PrismConfig defaults = PrismConfig.fromSettings(EpsilonSettings.defaults());
    Assert.assertEquals(defaults.inference(), InferencePath.HIP);
    PrismConfig eager = config("inference", "EAGER");
    eager.validate();
    Assert.assertEquals(eager.inference(), InferencePath.EAGER);
    Assert.assertEquals(PrismConfig.read(new Gson().toJson(eager)), eager);
    Assert.assertThrows(IllegalArgumentException.class, () -> config("inference", "hip"));
    Assert.assertThrows(IllegalArgumentException.class, () -> config("inference", "CUDA"));
  }

  /** 網の形と補助の頭の重み: 既定は深さ 2・前線なし・補助 0.01、depth と frontier で形を変える。範囲の外の深さと負の重みは拒む。 */
  @Test
  public void shapeAndAuxiliaryWeightsFollowTheSettings() {
    PrismConfig defaults = PrismConfig.fromSettings(EpsilonSettings.defaults());
    Assert.assertEquals(defaults.shape(), PrismShape.standard(2));
    Assert.assertEquals(
        config("depth", "3", "frontier", "true").shape(), PrismShape.standard(3).withFrontier(true));
    Assert.assertEquals(defaults.auxiliaryLoss().resultWeight(), 0.01f);
    Assert.assertEquals(defaults.auxiliaryLoss().oracleWeight(), 0.01f);
    Assert.assertEquals(config("auxOracle", "0.002").auxiliaryLoss().oracleWeight(), 0.002f);
    for (String[] setting : new String[][] {{"depth", "4"}, {"auxOracle", "-0.1"}, {"auxResult", "-0.1"}}) {
      PrismConfig wrong = config(setting);
      Assert.assertThrows(IllegalArgumentException.class, wrong::validate);
    }
  }

  /**
   * TEMPO: ゲインは K0 から (1 + (t − t0)/τ)^(−p) で減り、学習率の倍率は min(1, K / 0.25)。K′ は R² = 0（版 1・補正なし）と NaN で K そのもの、
   * R² > 0 で K / (K + (1 − K)(1 − R²))。範囲の外のゲインと R² は拒む。
   */
  @Test
  public void filterGainDecaysFromTheStartIteration() {
    PrismConfig filter = config("filterFrom", "49", "filterGain", "0.125", "filterTau", "24", "filterPower", "0.5");
    filter.validate();
    Assert.assertEquals(filter.filterGain(10), 0.125, 1e-12);
    Assert.assertEquals(filter.filterGain(49), 0.125, 1e-12);
    Assert.assertEquals(filter.filterGain(49 + 72), 0.0625, 1e-12);
    Assert.assertEquals(filter.filterLrScale(49, 0), 0.5, 1e-12);
    Assert.assertThrows(IllegalArgumentException.class, () -> config("filterGain", "1").validate());
    PrismConfig lineage = config("filterGain", "0.9", "filterTau", "4");
    double gain = lineage.filterGain(10);
    Assert.assertEquals(lineage.filterGain(10, 0), gain);
    Assert.assertEquals(lineage.filterGain(10, Double.NaN), gain);
    Assert.assertEquals(lineage.filterGain(10, 0.22), gain / (gain + (1 - gain) * 0.78), 1e-12);
    Assert.assertTrue(lineage.filterGain(10, 0.22) > gain && lineage.filterGain(10, 1) == 1);
    Assert.assertThrows(IllegalArgumentException.class, () -> lineage.filterGain(10, 1.5));
  }

  /** TEMPO の学習率の倍率: GAIN は min(1, K′_t / 0.25)、FULL は K′_t の予定を変えずに倍率だけ 1。名前の間違いは拒む。 */
  @Test
  public void filterLrModeFullHoldsTheLearningRateAtOne() {
    String[] tempo = {"filterGain", "0.9", "filterTau", "4"};
    PrismConfig gain = config(tempo);
    PrismConfig full = config(with(tempo, "filterLrMode", "FULL"));
    full.validate();
    Assert.assertEquals(full.filterLrMode(), FilterLrMode.FULL);
    Assert.assertEquals(PrismConfig.read(new Gson().toJson(full)), full);
    // t = 76・R² = 0.225: K′ ≈ 0.154 で、GAIN の倍率は K′ / 0.25 ≈ 0.61、FULL は 1。K′ は同じ
    double kPrime = gain.filterGain(76, 0.225);
    Assert.assertEquals(kPrime, 0.154, 0.001);
    Assert.assertEquals(full.filterGain(76, 0.225), kPrime);
    Assert.assertEquals(gain.filterLrScale(76, 0.225), kPrime / PrismConfig.FILTER_LR_GAIN, 1e-12);
    Assert.assertEquals(full.filterLrScale(76, 0.225), 1.0);
    Assert.assertEquals(full.filterLrScale(1, 0), 1.0);
    Assert.assertEquals(gain.filterLrScale(1, 0), 1.0);
    Assert.assertThrows(IllegalArgumentException.class, () -> config(with(tempo, "filterLrMode", "full")));
    Assert.assertThrows(IllegalArgumentException.class, () -> config(with(tempo, "filterLrMode", "1")));
  }

  /** 半荘の数の段: 空なら gamesPerCohort、段は始めが t 以下で最も後の数、形と昇順の誤りは拒む。 */
  @Test
  public void gamesScheduleStepsTheCohortSize() {
    PrismConfig constant = config("gamesSchedule", "");
    Assert.assertEquals(constant.gamesPerCohort(1), constant.gamesPerCohort());
    PrismConfig stepped = PrismConfig.fromSettings(EpsilonSettings.defaults());
    stepped.validate();
    Assert.assertEquals(stepped.gamesSchedule(), "1:8192,9:16384,17:32768,25:65536");
    Assert.assertEquals(stepped.gamesPerCohort(1), 8192);
    Assert.assertEquals(stepped.gamesPerCohort(8), 8192);
    Assert.assertEquals(stepped.gamesPerCohort(9), 16384);
    Assert.assertEquals(stepped.gamesPerCohort(24), 32768);
    Assert.assertEquals(stepped.gamesPerCohort(100), 65536);
    Assert.assertEquals(config("gamesSchedule", "5:1024").gamesPerCohort(4), constant.gamesPerCohort());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("gamesSchedule", "9:1,1:2").validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("gamesSchedule", "1:0").validate());
    Assert.assertThrows(IllegalArgumentException.class, () -> config("gamesSchedule", "8192").validate());
  }

  /** pairs に鍵と値の組 more を足した列（後の組が前の同じ鍵を上書きする）。 */
  private static String[] with(String[] pairs, String... more) {
    String[] result = Arrays.copyOf(pairs, pairs.length + more.length);
    System.arraycopy(more, 0, result, pairs.length, more.length);
    return result;
  }

  /** 類ごとのロールインの率: 書いた類だけ上書きし、書かない類は打牌か確約の率。 */
  @Test
  public void rollinClassRatesOverrideOnlyTheNamedClasses() {
    var sampling = config("rollinClassRates", "CALL=0.15").sampling();
    Assert.assertEquals(sampling.rollinRate(DecisionClass.CALL), 0.15);
    Assert.assertEquals(sampling.rollinRate(DecisionClass.THREAT), sampling.rollinRateCommit());
    Assert.assertEquals(sampling.rollinRate(DecisionClass.DISCARD_MID), sampling.rollinRateDiscard());
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
}
