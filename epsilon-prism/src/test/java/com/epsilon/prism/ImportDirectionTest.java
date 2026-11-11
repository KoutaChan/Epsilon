package com.epsilon.prism;

import java.io.IOException;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * prism の依存の向きを確かめる。
 *
 * <ul>
 *   <li>main とテストは classic・prism-4p を import せず、ワイルドカードの import も使わない。
 *   <li>pom では classic を runtime の scope にだけ置く。
 *   <li>prism-4p のクラスは、テストの class path に無い。運のネットワークのサンプルを受け取るフックと受け取り先（LuckHooks・LuckSink）も
 *       無い。
 *   <li>運の精算（luck）は collect・learn・run・app を読まず、rows と noise は luck を読まない。
 *   <li>運の精算のクラスは luck にまとめたので、前に置いていたパッケージには残っていない。
 * </ul>
 */
public class ImportDirectionTest {

  @Test
  public void importsNoClassicNorTheVariantModule() throws IOException {
    Assert.assertEquals(
        ImportDirection.violations(ImportDirection.CLASSIC_PACKAGES + "|com\\.epsilon\\.prism4p(\\..*)?"), List.of());
  }

  @Test
  public void pomNamesClassicOnlyAtRuntime() throws IOException {
    Assert.assertEquals(ImportDirection.pomViolations(), List.of());
  }

  @Test
  public void classpathHoldsNoVariantModule() {
    Assert.assertEquals(ImportDirection.loadable(List.of("com.epsilon.prism4p.variant.FourPlayer")), List.of());
  }

  /** 運の精算（luck）は、収集・学習・系譜・コマンド（collect・learn・run・app）を読まない。 */
  @Test
  public void luckReadsNoCollectLearnRunNorApp() throws IOException {
    Assert.assertEquals(
        ImportDirection.packageViolations("com.epsilon.prism.luck", "com\\.epsilon\\.prism\\.(collect|learn|run|app)(\\..*)?"),
        List.of());
  }

  /** 行（rows）と、ラベルのノイズの対策（noise）は、運の精算（luck）を読まない。 */
  @Test
  public void rowsAndNoiseReadNoLuck() throws IOException {
    for (String packageName : List.of("com.epsilon.prism.rows", "com.epsilon.prism.noise"))
      Assert.assertEquals(
          ImportDirection.packageViolations(packageName, "com\\.epsilon\\.prism\\.luck(\\..*)?"), List.of(), packageName);
  }

  /** 運の精算のクラスは luck にまとめたので、前に置いていたパッケージ（collect・learn・rows・label）のクラスは class path に無い。 */
  @Test
  public void classpathHoldsNoFormerLedgerClasses() {
    Assert.assertEquals(
        ImportDirection.loadable(
            List.of(
                "com.epsilon.prism.collect.EventLedger",
                "com.epsilon.prism.collect.ChanceBatch",
                "com.epsilon.prism.learn.LedgerFit",
                "com.epsilon.prism.rows.LedgerRow",
                "com.epsilon.prism.label.UraDistribution",
                "com.epsilon.prism.label.HandEndValues",
                "com.epsilon.prism.label.UraAudit",
                "com.epsilon.prism.label.UraMode")),
        List.of());
  }

  @Test
  public void classpathHoldsNoLuckSamplePorts() {
    Assert.assertEquals(
        ImportDirection.loadable(List.of("com.epsilon.prism.collect.LuckHooks", "com.epsilon.prism.collect.LuckSink")),
        List.of());
  }
}
