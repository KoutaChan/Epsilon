package com.epsilon.grp.rules;

import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.mahjong.rules.Variant;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * テスト用に乱数で打った半荘から作る数のファイルは、中身が決まったバイト列になる（数のファイルの形が変われば、ここで分かる）。
 * 期待する値は、テスト用の Φ（GrpFixtures.STAMP）をヘッダーに書いて同じ半荘から求めた、局の数と、gzip を解いた中身の SHA-256。
 */
public class MovedCountFileTest {

  /** ルール、乱数の種、局の数、gzip を解いた中身の SHA-256。 */
  private static final Object[][] EXPECTED = {
    {Variant.TENHOU_4P, 71L, 457, "7414985e4f750dfaeec13b6f77b83e45cb4be6251545188f45bc516c611abac4"},
    {Variant.TENHOU_4P, 7L, 491, "acadd47caf8268b8f5e84fc941b4c88c20b5382635769812041557f52cae3a76"},
    {Variant.TENHOU_3P, 71L, 433, "697c4af90062b96f89895353d48a1dceb448a600160a0a7386615fcf75eaab8c"},
    {Variant.TENHOU_3P, 7L, 473, "8cf8e1cffea00dbcbdba26349ad1f7423179ea7441e35dc59a308fe1514a9599"},
  };

  @Test
  public void playedCountsKeepTheirBytes() throws Exception {
    Path directory = Files.createTempDirectory("moved-counts");
    for (Object[] row : EXPECTED) {
      Variant variant = (Variant) row[0];
      CountedHands counts = RulesFixtures.play(variant, 50, 100, (Long) row[1]).counts();
      String context = variant + " seed " + row[1];
      Assert.assertEquals(counts.hands(), (int) (Integer) row[2], context);
      Path file = directory.resolve(variant + "-" + row[1] + ".bin");
      CountFile.write(file, counts);
      Assert.assertEquals(contentSha256(file), row[3], context);
    }
  }

  /** 数のファイルの、gzip を解いた中身の SHA-256（小文字の 16 進）。 */
  private static String contentSha256(Path file) throws Exception {
    try (InputStream in = new GZIPInputStream(Files.newInputStream(file))) {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(in.readAllBytes()));
    }
  }
}
