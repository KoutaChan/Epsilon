package com.epsilon.prism.grp;

import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.Phi;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 反復ごとに Φ を決めるテスト。点数の順位から始め、READY の候補が効く反復で Phi を開き直す。同じ項目が効く間は、同じ Phi を使う。再開
 * しても同じ反復には同じ項目が効き、値はネットワークの Φ とビットで同じ。
 */
public class PhiResolverTest {

  @Test
  public void candidatesRebuildThePhi() throws Exception {
    Path run = Files.createTempDirectory("prism-resolver");
    try {
      var files = new GrpFiles(run);
      GrpFixtures.writeState(run, 1);
      GrpState probe = GrpFixtures.games(0, 8, 3).get(5).handEnds().get(1);
      Path a = GrpFixtures.network(run.resolve("a"), 11);
      Path b = GrpFixtures.network(run.resolve("b"), 13);
      double[] expectedA = direct(a, probe);
      double[] expectedB = direct(b, probe);
      PhiSettings phi = GrpFixtures.scoreStart().phi();
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), files)) {
        ActivePhi first = resolver.resolve(1, phi);
        Assert.assertEquals(first.entry().kind(), PhiEntryKind.SCORE_RANK);
        Assert.assertEquals(first.phi().kind(), Phi.Kind.SCORE_RANK);
        PhiHistory.append(files, GrpFixtures.ready(files, "g001-a1", 1, a).entry(GrpFixtures.VARIANT, 3, "test"));
        PhiHistory.append(files, GrpFixtures.ready(files, "g003-a1", 3, b).entry(GrpFixtures.VARIANT, 5, "test"));
        PhiHistory.append(files, PhiHistory.read(files).getFirst().from(6, PhiEntry.ROLLBACK + " test"));
        Assert.assertSame(resolver.resolve(2, phi).phi(), first.phi());
        ActivePhi third = resolver.resolve(3, phi);
        Assert.assertEquals(third.entry().id(), "g001-a1");
        Assert.assertEquals(bits(values(third.phi(), probe)), bits(expectedA));
        ActivePhi fourth = resolver.resolve(4, phi);
        Assert.assertSame(fourth.phi(), third.phi());
        ActivePhi fifth = resolver.resolve(5, phi);
        Assert.assertNotSame(fifth.phi(), third.phi());
        Assert.assertEquals(bits(values(fifth.phi(), probe)), bits(expectedB));
        ActivePhi sixth = resolver.resolve(6, phi);
        Assert.assertEquals(sixth.entry().kind(), PhiEntryKind.SCORE_RANK);
      }
      // 再開しても、同じ反復には同じ項目が効く
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), files)) {
        ActivePhi fourth = resolver.resolve(4, phi);
        Assert.assertEquals(fourth.entry().id(), "g001-a1");
        Assert.assertEquals(bits(values(fourth.phi(), probe)), bits(expectedA));
        Assert.assertEquals(resolver.resolve(3, phi).entry().id(), "g001-a1");
        Assert.assertEquals(resolver.resolve(7, phi).entry().kind(), PhiEntryKind.SCORE_RANK);
      }
      Assert.assertEquals(RefitRecord.readAll(files).size(), 2, "nothing was rejected");
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** チェックポイントのネットワークに局面を直に通して求めた Φ。 */
  static double[] direct(Path checkpoint, GrpState state) throws java.io.IOException {
    GrpNetwork network = (GrpNetwork) GrpCheckpoint.read(checkpoint);
    double[] values = new double[4];
    network.phi(state, GrpFixtures.UTILITY, GrpWorkspace.of(network.shape()), values);
    return values;
  }

  static double[] values(Phi phi, GrpState state) {
    double[] values = new double[4];
    phi.values(state, values);
    return values;
  }

  static long[] bits(double[] values) {
    return Arrays.stream(values).mapToLong(Double::doubleToRawLongBits).toArray();
  }
}
