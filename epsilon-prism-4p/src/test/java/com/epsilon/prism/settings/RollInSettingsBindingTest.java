package com.epsilon.prism.settings;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Change;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.collect.RollInRule;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import com.epsilon.prism.core.Openness;
import com.epsilon.prism.noise.RateSettings;
import com.epsilon.prism.run.Snapshot;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 写しの決まりの上書き（labels.copies）と副露の率（noise.rates.open）を設定から読むテスト。書いたキーだけがセクションの値に替わり、
 * 範囲外の値と知らないクラスは拒む。続いている run で settings change が残す行も確かめる。
 */
public class RollInSettingsBindingTest {

  /** 既定では上書きがなく、逸脱は ANY で、全ての根のクラスがセクションの値を使う。 */
  @Test
  public void defaultsHaveNoOverrides() {
    PrismSettings settings = TestSettings.of();
    LabelSettings labels = settings.labels();
    Assert.assertTrue(labels.copies().isEmpty());
    Assert.assertEquals(labels.deviation(), DeviationTarget.ANY);
    Assert.assertTrue(settings.noise().rates().open().isEmpty());
    for (DecisionClass root : ClassTables.classes(settings.run().variant())) {
      Assert.assertEquals(labels.deviationFloor(root), 0.1);
      Assert.assertEquals(labels.deviation(root), DeviationTarget.ANY);
      Assert.assertEquals(labels.maxCopyLabels(root), 2);
      for (DecisionClass later : DecisionClass.values()) Assert.assertEquals(labels.copyLabelRate(root, later), 0.3);
      for (Openness hand : Openness.values())
        Assert.assertEquals(settings.noise().rates().rate(root, hand), settings.noise().rates().rate(root));
    }
  }

  /**
   * 上書きは書いたキーにだけ効く。根のクラスの上書きの中では、後の判断のクラスごとの ρ_br が、その根のクラスの ρ_br より優先する。
   * 書かないキーはセクションの値を使い、セクションの値を変えると、上書きしていない根のクラスとキーはそれに従う。収集で使う率も同じ値になる。
   */
  @Test
  public void overridesReplaceOnlyTheirKeys() {
    PrismSettings settings =
        TestSettings.of(
            "labels.copyLabelRate=0.25",
            "labels.deviationFloor=0.2",
            "labels.copies.CALL.copyLabelRate=0.6",
            "labels.copies.CALL.laterRates.DISCARD_EARLY=0.05",
            "labels.copies.CALL.deviation=OTHER",
            "labels.copies.RIICHI.maxCopyLabels=5",
            "labels.copies.RIICHI.deviationFloor=0.5",
            "labels.copies.KAN.laterRates.THREAT=0.9");
    LabelSettings labels = settings.labels();
    Assert.assertEquals(labels.copies().keySet(), EnumSet.of(DecisionClass.RIICHI, DecisionClass.CALL, DecisionClass.KAN));
    Assert.assertEquals(labels.copyLabelRate(DecisionClass.CALL, DecisionClass.DISCARD_EARLY), 0.05);
    Assert.assertEquals(labels.copyLabelRate(DecisionClass.CALL, DecisionClass.THREAT), 0.6);
    Assert.assertEquals(labels.copyLabelRate(DecisionClass.RIICHI, DecisionClass.THREAT), 0.25);
    Assert.assertEquals(labels.copyLabelRate(DecisionClass.KAN, DecisionClass.THREAT), 0.9);
    Assert.assertEquals(labels.copyLabelRate(DecisionClass.KAN, DecisionClass.CALL), 0.25);
    Assert.assertEquals(labels.copyLabelRate(DecisionClass.THREAT, DecisionClass.CALL), 0.25);
    Assert.assertEquals(labels.deviation(DecisionClass.CALL), DeviationTarget.OTHER);
    Assert.assertEquals(labels.deviation(DecisionClass.RIICHI), DeviationTarget.ANY);
    Assert.assertEquals(labels.deviationFloor(DecisionClass.RIICHI), 0.5);
    Assert.assertEquals(labels.deviationFloor(DecisionClass.CALL), 0.2);
    Assert.assertEquals(labels.maxCopyLabels(DecisionClass.RIICHI), 5);
    Assert.assertEquals(labels.maxCopyLabels(DecisionClass.CALL), 2);

    Sampling sampling = Sampling.of(settings);
    RollInRule call = sampling.rollIn(DecisionClass.CALL);
    Assert.assertEquals(call.rate(), 0.03);
    Assert.assertEquals(call.target(), DeviationTarget.OTHER);
    Assert.assertEquals(call.epsilonQ(), 0.2);
    Assert.assertEquals(call.laterRate(DecisionClass.DISCARD_EARLY), 0.05);
    Assert.assertEquals(call.laterRate(DecisionClass.DISCARD_MID), 0.6);
    Assert.assertEquals(call.maxLabels(), 2);
    RollInRule riichi = sampling.rollIn(DecisionClass.RIICHI);
    Assert.assertEquals(riichi.epsilonQ(), 0.5);
    Assert.assertEquals(riichi.maxLabels(), 5);
    Assert.assertEquals(riichi.target(), DeviationTarget.ANY);
    Assert.assertEquals(sampling.rollIn(DecisionClass.THREAT).laterRate(DecisionClass.CALL), 0.25);
    Assert.assertEquals(sampling.rollIn(DecisionClass.DISCARD_EARLY).rate(), 0.002);

    // セクションの deviation は、上書きしていない根のクラスに効く
    LabelSettings other = TestSettings.of("labels.deviation=OTHER", "labels.copies.CALL.deviation=ANY").labels();
    Assert.assertEquals(other.deviation(DecisionClass.THREAT), DeviationTarget.OTHER);
    Assert.assertEquals(other.deviation(DecisionClass.CALL), DeviationTarget.ANY);
  }

  /**
   * 副露の率は副露している席にだけ効き、書かないクラスは率の表の率を使う。どちらも倍率を掛け、1 を上限に切る。1 でない倍率は今の jar の
   * 設定では拒むので、セクションを直に作る。
   */
  @Test
  public void openRatesUseTheMultiplier() {
    RateSettings bound = TestSettings.of().noise().rates();
    RateSettings rates =
        new RateSettings(
            bound.table(),
            Map.of(DecisionClass.CALL, 0.4, DecisionClass.DISCARD_MID, 0.7),
            2,
            bound.allActions(),
            bound.review());
    Assert.assertEquals(rates.rate(DecisionClass.CALL, Openness.OPEN), 0.8);
    Assert.assertEquals(rates.rate(DecisionClass.CALL, Openness.CLOSED), 0.208);
    Assert.assertEquals(rates.rate(DecisionClass.DISCARD_MID, Openness.OPEN), 1.0);
    Assert.assertEquals(rates.rate(DecisionClass.DISCARD_MID, Openness.CLOSED), 0.078);
    Assert.assertEquals(rates.rate(DecisionClass.THREAT, Openness.OPEN), 0.208);
    Assert.assertEquals(rates.rate(DecisionClass.FINAL_END, Openness.OPEN), 1.0);
    Sampling sampling = Sampling.of(TestSettings.of("noise.rates.open.CALL=0.4"));
    Assert.assertEquals(sampling.labelRate(DecisionClass.CALL, Openness.OPEN), 0.4);
    Assert.assertEquals(sampling.labelRate(DecisionClass.CALL, Openness.CLOSED), 0.104);
    Assert.assertEquals(sampling.labelRate(DecisionClass.KAN, Openness.OPEN), 0.104);
  }

  /** 範囲外の率と上限、知らないクラス、知らない逸脱の種類は拒む。 */
  @Test
  public void invalidValuesAreRefused() {
    for (String assignment :
        List.of(
            "labels.copies.CALL.copyLabelRate=1.5",
            "labels.copies.CALL.deviationFloor=-0.1",
            "labels.copies.CALL.maxCopyLabels=-1",
            "labels.copies.CALL.laterRates.THREAT=2",
            "labels.copies.CALL.laterRates.NOPE=0.1",
            "labels.copies.NOPE.copyLabelRate=0.1",
            "labels.copies.CALL.deviation=SOME",
            "labels.copies.CALL.rate=0.1",
            "labels.deviation=SOME",
            "noise.rates.open.CALL=1.5",
            "noise.rates.open.CALL=-0.1",
            "noise.rates.open.NOPE=0.1"))
      Assert.expectThrows(IllegalArgumentException.class, () -> TestSettings.of(assignment));
  }

  /**
   * 続いている run に、鳴ける判断の β を 0.15 に、その写しの ρ_br を 0.6 に、上限を 4 にするファイルを重ねると、変えた 3 つのキーが
   * COLLECT として設定の行に残り、次の反復から効く。同じ変更をもう 1 度重ねても行は増えない。副露の率の変更も COLLECT になる。
   */
  @Test
  public void settingsChangeRecordsTheKeysAsCollect() throws Exception {
    Path run = Files.createTempDirectory("prism-rollin-change");
    try {
      TestSettings.create(run, TestSettings.cpuLineage());
      Path apply = run.resolve("call-arm.toml");
      Files.writeString(
          apply,
          """
          [labels.rollin]
          CALL = 0.15

          [labels.copies.CALL]
          copyLabelRate = 0.6
          maxCopyLabels = 4
          """,
          StandardCharsets.UTF_8);
      Assert.assertEquals(apply(run, apply), Exit.OK.code());
      SettingsRow row = rows(run).getLast();
      Assert.assertEquals(row.row(), 2);
      Assert.assertEquals(row.from(), 1);
      Map<String, Change> changes =
          row.changes().stream().collect(Collectors.toMap(SettingsRow.ChangedKey::key, SettingsRow.ChangedKey::change));
      Assert.assertEquals(
          changes,
          Map.of(
              "labels.rollin.CALL", Change.COLLECT,
              "labels.copies.CALL.copyLabelRate", Change.COLLECT,
              "labels.copies.CALL.maxCopyLabels", Change.COLLECT),
          changes.toString());
      PrismSettings latest = row.bind(PrismSettings.class).value();
      Assert.assertEquals(Sampling.of(latest).rollIn(DecisionClass.CALL).rate(), 0.15);
      Assert.assertEquals(apply(run, apply), Exit.OK.code());
      Assert.assertEquals(rows(run).size(), 2);

      Assert.assertEquals(
          Main.run(
              new String[] {
                "settings", "change", "--run", run.toString(), "--change", "noise.rates.open.CALL=0.5",
                "--change", "labels.copies.THREAT.deviation=OTHER", "--reason", "test"
              }),
          Exit.OK.code());
      SettingsRow third = rows(run).getLast();
      Assert.assertEquals(third.row(), 3);
      Assert.assertEquals(
          third.changes().stream().collect(Collectors.toMap(SettingsRow.ChangedKey::key, SettingsRow.ChangedKey::change)),
          Map.of("noise.rates.open.CALL", Change.COLLECT, "labels.copies.THREAT.deviation", Change.COLLECT));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  private static int apply(Path run, Path file) {
    return Main.run(
        new String[] {"settings", "change", "--run", run.toString(), "--apply", file.toString(), "--reason", "test"});
  }

  private static List<SettingsRow> rows(Path run) throws Exception {
    return new SettingsRows(new RunDirectory(run)).read();
  }
}
