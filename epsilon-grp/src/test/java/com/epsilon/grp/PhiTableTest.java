package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * PhiTable を確かめる。温度（行の和、T = 1、表より先の局）、補正（区間の中心の間の線形補間、端の値、席の和、表より先の局）、空の表、
 * JSON の往復、拒む表。
 */
public class PhiTableTest {

  private static final double TOLERANCE = 1e-12;

  private static final double[] MARGINALS = {
    0.70, 0.20, 0.08, 0.02,
    0.20, 0.50, 0.20, 0.10,
    0.08, 0.20, 0.50, 0.22,
    0.02, 0.10, 0.22, 0.66
  };

  @Test
  public void temperaturesKeepRowSumsAndUseTheLastValueLater() {
    PhiTable table = PhiTable.of(Variant.TENHOU_4P, new double[] {1, 2}, 0, 1, new double[0][]);
    double[] marginals = MARGINALS.clone();
    table.temper(0, marginals);
    Assert.assertEquals(marginals, MARGINALS);
    table.temper(9, marginals);
    for (int seat = 0; seat < 4; seat++) {
      double total = 0;
      for (int rank = 0; rank < 4; rank++) {
        total += marginals[seat * 4 + rank];
      }
      Assert.assertEquals(total, 1, TOLERANCE);
    }
    double softened = Math.sqrt(0.70) / (Math.sqrt(0.70) + Math.sqrt(0.20) + Math.sqrt(0.08) + Math.sqrt(0.02));
    Assert.assertEquals(marginals[0], softened, TOLERANCE);
  }

  @Test
  public void correctionsInterpolateBetweenBinCentersAndKeepTheSum() {
    // 区間は [0, 0.5) と [0.5, 1] で中心は 0.25 と 0.75。c(0.5) = 0.5、0.25 以下は 0、0.75 以上は 1。
    PhiTable table = PhiTable.of(Variant.TENHOU_4P, new double[0], 0, 1, new double[][] {{0, 1}});
    double[] phi = {0.5, 0.25, 0.75, 0};
    table.calibrate(3, phi);
    double mean = (0.5 + 0 + 1 + 0) / 4;
    Assert.assertEquals(phi[0], 0.5 + 0.5 - mean, TOLERANCE);
    Assert.assertEquals(phi[1], 0.25 - mean, TOLERANCE);
    Assert.assertEquals(phi[2], 0.75 + 1 - mean, TOLERANCE);
    Assert.assertEquals(phi[3], -mean, TOLERANCE);
    Assert.assertEquals(phi[0] + phi[1] + phi[2] + phi[3], 1.5, TOLERANCE);
  }

  @Test
  public void laterRoundsUseTheLastStage() {
    PhiTable table = PhiTable.of(Variant.TENHOU_3P, new double[0], -1, 1, new double[][] {{0, 0}, {0.3, -0.3}});
    double[] early = {0.8, 0.1, -0.9};
    table.calibrate(0, early);
    Assert.assertEquals(early, new double[] {0.8, 0.1, -0.9});
    double[] late = {0.8, 0.1, -0.9};
    table.calibrate(8, late);
    Assert.assertEquals(late[0] + late[1] + late[2], 0.0, TOLERANCE);
    Assert.assertTrue(late[2] > -0.9, Double.toString(late[2]));
  }

  @Test
  public void emptyTablesChangeNothing() {
    PhiTable table = PhiTable.of(Variant.TENHOU_4P, new double[0], 0, 0, new double[0][]);
    double[] marginals = MARGINALS.clone();
    table.temper(5, marginals);
    Assert.assertEquals(marginals, MARGINALS);
    double[] phi = {0.6, 0.2, 0.1, -0.9};
    table.calibrate(5, phi);
    Assert.assertEquals(phi, new double[] {0.6, 0.2, 0.1, -0.9});
  }

  @Test
  public void tablesRoundTripThroughJson() throws IOException {
    PhiTable table =
        PhiTable.of(Variant.TENHOU_4P, new double[] {1.07, 1.0 / 3}, -1, 2.0 / 3, new double[][] {{0.01, -0.02, 0.1 / 3}});
    Path file = Files.createTempFile("phi-table", ".json");
    try {
      table.write(file);
      PhiTable read = PhiTable.read(file);
      Assert.assertEquals(read.variant(), Variant.TENHOU_4P);
      double[] expected = MARGINALS.clone();
      double[] actual = MARGINALS.clone();
      table.temper(1, expected);
      read.temper(1, actual);
      Assert.assertEquals(actual, expected);
      double[] expectedPhi = {0.6, 0.2, 0.1, -0.9};
      double[] actualPhi = expectedPhi.clone();
      table.calibrate(2, expectedPhi);
      read.calibrate(2, actualPhi);
      Assert.assertEquals(actualPhi, expectedPhi);
    } finally {
      Files.delete(file);
    }
  }

  /** 書き直しは一時ファイルを残さずにファイルを置き換え、SHA-256 は中身のバイト列から求めた値になる。 */
  @Test
  public void rewritesReplaceTheFileAndShaFollowsTheBytes() throws Exception {
    Path directory = Files.createTempDirectory("phi-table");
    Path file = directory.resolve("table.json");
    try {
      PhiTable.of(Variant.TENHOU_4P, new double[] {1.1}, -1, 2.0 / 3, new double[][] {{0.01}}).write(file);
      String first = PhiTable.sha256(file);
      PhiTable.of(Variant.TENHOU_4P, new double[] {0.9}, -1, 2.0 / 3, new double[][] {{0.02}}).write(file);
      String second = PhiTable.sha256(file);
      Assert.assertNotEquals(second, first);
      Assert.assertEquals(
          second,
          HexFormat.of().formatHex(
              MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file))));
      Assert.assertEquals(PhiTable.read(file).variant(), Variant.TENHOU_4P);
      try (var entries = Files.list(directory)) {
        Assert.assertEquals(entries.map(path -> path.getFileName().toString()).toList(), List.of("table.json"));
      }
    } finally {
      Files.deleteIfExists(file);
      Files.delete(directory);
    }
  }

  @Test
  public void invalidTablesAreRefused() {
    Assert.assertThrows(
        IllegalArgumentException.class, () -> PhiTable.of(Variant.TENHOU_4P, new double[] {0}, 0, 1, new double[0][]));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> PhiTable.of(Variant.TENHOU_4P, new double[0], 1, 0, new double[][] {{0}}));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> PhiTable.of(Variant.TENHOU_4P, new double[0], 0, 1, new double[][] {{0, 1}, {0}}));
    PhiTable table = PhiTable.of(Variant.TENHOU_4P, new double[] {2}, 0, 1, new double[][] {{0}});
    Assert.assertThrows(IllegalArgumentException.class, () -> table.temper(12, MARGINALS.clone()));
    Assert.assertThrows(IllegalArgumentException.class, () -> table.calibrate(0, new double[3]));
  }
}
