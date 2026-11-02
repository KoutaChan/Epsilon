package com.epsilon.prism.grp;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.config.settings.GrpInferenceSettings;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 反復ごとの Φ の解決（CPU の GRP）: 点数の順位で始め、READY の候補が効く反復で GrpPhi を作り直し、Φ の種類が点数だけから GRP に変わる反復だけ
 * 運の精算の R² を 0 にする。同じ項目のあいだは同じ値を使い、GRP どうし・巻き戻しでは R² を持ち越す。再開しても同じ項目と同じ R² の扱いになり、
 * 値はラベルと同じ maxBatch 行の順伝播とビットで同じ。開いているあいだは学習の JVM が動いているとみなす。
 */
public class PhiResolverTest {
  static final int MAX_BATCH = EpsilonSettings.defaults().bind(GrpInferenceSettings.class).maxBatch();

  /** checkpoint の GRP を maxBatch 行ずつ直に通した Φ（閉じると GRP も閉じる）。 */
  static HandEndPhi direct(Path checkpoint) throws IOException {
    Model model = EpsilonGrpCheckpointManager.load(checkpoint, Device.cpu());
    HandEndPhi phi = GrpFixtures.direct(new EpsilonGrpInference(model, MAX_BATCH));
    return new HandEndPhi() {
      @Override
      public PhiValues values(List<HandEndInput> inputs) throws IOException {
        return phi.values(inputs);
      }

      @Override
      public void close() {
        model.close();
      }
    };
  }

  /** ドライバのスレッドで 1 回評価する。 */
  static double[] request(HandValue value, ExecutorService driver, HandEndInput input) throws Exception {
    var result = new CompletableFuture<double[]>();
    driver.execute(
        () -> {
          try {
            value.request(input.prefix(), input.next().scores(), result::complete);
          } catch (RuntimeException failure) {
            result.completeExceptionally(failure);
          }
        });
    return result.get(2, TimeUnit.MINUTES);
  }

  @Test(timeOut = 600_000)
  public void candidatesRebuildThePhiAndResetTheLedgerOnlyFromAScoreOnlyPhi() throws Exception {
    Path run = Files.createTempDirectory("prism-resolver");
    ExecutorService driver = Executors.newSingleThreadExecutor();
    try {
      var files = new GrpFiles(run);
      GrpFixtures.writeState(run, 1);
      List<RecordedGame> games = GrpFixtures.games(0, 64, 3);
      HandEndInput probe = games.get(5).input(2);
      Path a = GrpFixtures.checkpoint(run.resolve("a"));
      Path b = GrpFixtures.checkpoint(run.resolve("b"));
      double[] expectedA, expectedB;
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), GrpFixtures.PROFILE, driver::execute, files)) {
        Assert.assertEquals(PhiHistory.firstOpenIteration(files), 2, "the training JVM holds iteration 1");
        ActivePhi first = resolver.resolve(1);
        Assert.assertEquals(first.entry().kind(), PhiEntryKind.SCORE_RANK);
        Assert.assertFalse(first.resetsLedger());
        Assert.assertEquals(request(first.value(), driver, probe), HandValue.terminal(probe.next().scores(), GrpFixtures.PROFILE));
        try (HandEndPhi phiA = direct(a);
            HandEndPhi phiB = direct(b)) {
          PhiHistory.append(files, GrpFixtures.ready(files, "g007-a1", a, phiA, games).entry(3, "test"));
          PhiHistory.append(files, GrpFixtures.ready(files, "g009-a1", b, phiB, games).entry(5, "test"));
          expectedA = phiA.values(List.of(probe)).utilities()[0];
          expectedB = phiB.values(List.of(probe)).utilities()[0];
        }
        PhiHistory.append(files, PhiHistory.read(files).getFirst().from(6, "rollback"));
        Assert.assertSame(resolver.resolve(2).value(), first.value());
        ActivePhi third = resolver.resolve(3);
        Assert.assertEquals(third.entry().id(), "g007-a1");
        Assert.assertTrue(third.resetsLedger(), "score-rank -> GRP resets the ledger R²");
        Assert.assertEquals(bits(request(third.value(), driver, probe)), bits(expectedA));
        ActivePhi fourth = resolver.resolve(4);
        Assert.assertSame(fourth.value(), third.value());
        Assert.assertFalse(fourth.resetsLedger());
        ActivePhi fifth = resolver.resolve(5);
        Assert.assertNotSame(fifth.value(), third.value());
        Assert.assertFalse(fifth.resetsLedger(), "GRP -> GRP keeps the ledger R²");
        Assert.assertEquals(bits(request(fifth.value(), driver, probe)), bits(expectedB));
        ActivePhi sixth = resolver.resolve(6);
        Assert.assertEquals(sixth.entry().kind(), PhiEntryKind.SCORE_RANK);
        Assert.assertFalse(sixth.resetsLedger());
      }
      Assert.assertEquals(PhiHistory.firstOpenIteration(files), 1);
      // 再開: 同じ反復には同じ項目と同じ R² の扱い
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), GrpFixtures.PROFILE, driver::execute, files)) {
        ActivePhi fourth = resolver.resolve(4);
        Assert.assertEquals(fourth.entry().id(), "g007-a1");
        Assert.assertFalse(fourth.resetsLedger());
        Assert.assertEquals(bits(request(fourth.value(), driver, probe)), bits(expectedA));
        Assert.assertTrue(resolver.resolve(3).resetsLedger());
        Assert.assertEquals(resolver.resolve(7).entry().kind(), PhiEntryKind.SCORE_RANK);
      }
      Assert.assertEquals(RefitRecord.readAll(files).size(), 2, "nothing was rejected");
    } finally {
      driver.shutdownNow();
      Snapshot.deleteTree(run);
    }
  }

  static long[] bits(double[] values) {
    return Arrays.stream(values).mapToLong(Double::doubleToRawLongBits).toArray();
  }
}
