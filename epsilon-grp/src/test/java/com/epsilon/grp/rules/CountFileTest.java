package com.epsilon.grp.rules;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.grp.train.PhiStamp;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 数のファイルと数の列を確かめる。確かめるのは、往復（半荘 ID の昇順に並べ替え、書き直すと同じバイト列になる）、壊れたファイル
 * （途中で切れた・後ろに余りがある・半荘の順が乱れた・局の無い半荘がある・点の和が合わない・知らないルール・圧縮が壊れた）を拒むこと、
 * 数えるときの検査、局の進みを再現して確かめること。ヘッダーの Φ（{@link PhiStamp}）が往復することと、壊れた Φ のフィールドを拒むことも
 * 確かめる。
 */
public class CountFileTest {

  private static final Variant FOUR = Variant.TENHOU_4P;

  @Test
  public void filesRoundTripByteForByte() throws IOException {
    Path directory = Files.createTempDirectory("count-file");
    for (Variant variant : Variant.values()) {
      CountedHands counts = RulesFixtures.play(variant, 50, 100, 71).counts();
      Assert.assertEquals(counts.firstMismatch(), -1, "played games must replay");
      Path file = directory.resolve(variant + ".bin");
      CountFile.write(file, counts);
      CountedHands read = CountFile.read(file);
      assertSame(read, counts);
      Path again = directory.resolve(variant + "-again.bin");
      CountFile.write(again, read);
      Assert.assertEquals(Files.readAllBytes(again), Files.readAllBytes(file));
      Assert.assertFalse(Files.exists(directory.resolve(variant + ".bin.tmp")));
    }
  }

  @Test
  public void gamesAreSortedById() {
    CountedHands.Builder shuffled = CountedHands.builder(FOUR, 7, 0.5, GrpFixtures.STAMP);
    CountedHands.Builder sorted = CountedHands.builder(FOUR, 7, 0.5, GrpFixtures.STAMP);
    CountedHands played = RulesFixtures.play(FOUR, 3, 1, 73).counts();
    for (int game : new int[] {2, 0, 1}) {
      add(shuffled, played, game, 30 - 10L * game);
    }
    for (int game : new int[] {2, 1, 0}) {
      add(sorted, played, game, 30 - 10L * game);
    }
    CountedHands a = shuffled.build();
    CountedHands b = sorted.build();
    Assert.assertEquals(a.gameId(0), 10L);
    Assert.assertEquals(a.gameId(2), 30L);
    assertSame(a, b);
    Assert.assertEquals(a.cohort(), 7);
    Assert.assertEquals(a.exploreTemperature(), 0.5);
  }

  @Test
  public void brokenFilesAreRejected() throws IOException {
    Path directory = Files.createTempDirectory("count-broken");
    CountedHands counts = RulesFixtures.play(FOUR, 20, 1, 79).counts();
    Path good = directory.resolve("good.bin");
    CountFile.write(good, counts);
    byte[] bytes = Files.readAllBytes(good);
    assertRejected(directory, "truncated.bin", Arrays.copyOf(bytes, bytes.length / 2));
    byte[] flipped = bytes.clone();
    flipped[flipped.length / 2] ^= 0x5A;
    assertRejected(directory, "flipped.bin", flipped);
    long code = RulesFixtures.allNoten(FOUR);
    int[] even = {25_000, 25_000, 25_000, 25_000};
    assertRejected(directory, "trailing.bin", gzip(out -> {
      header(out, FOUR.name(), 1);
      game(out, 5, 1);
      hand(out, 0, 0, 0, even, code);
      out.writeByte(0);
    }));
    assertRejected(directory, "order.bin", gzip(out -> {
      header(out, FOUR.name(), 2);
      game(out, 5, 1);
      hand(out, 0, 0, 0, even, code);
      game(out, 5, 1);
      hand(out, 0, 0, 0, even, code);
    }));
    assertRejected(directory, "empty-game.bin", gzip(out -> {
      header(out, FOUR.name(), 1);
      game(out, 5, 0);
    }));
    assertRejected(directory, "sum.bin", gzip(out -> {
      header(out, FOUR.name(), 1);
      game(out, 5, 1);
      hand(out, 0, 0, 0, new int[] {25_000, 25_000, 25_000, 24_000}, code);
    }));
    assertRejected(directory, "variant.bin", gzip(out -> {
      header(out, "TENHOU_5P", 0);
    }));
    assertRejected(directory, "code.bin", gzip(out -> {
      header(out, FOUR.name(), 1);
      game(out, 5, 1);
      hand(out, 0, 0, 0, even, -1L);
    }));
    assertRejected(directory, "short.bin", gzip(out -> {
      header(out, FOUR.name(), 2);
      game(out, 5, 1);
      hand(out, 0, 0, 0, even, code);
    }));
    // 手で書いた正しいファイルは読める（上で拒んだのが、形の誤りだけのせいだと確かめる）
    Path written = directory.resolve("handmade.bin");
    Files.write(written, gzip(out -> {
      header(out, FOUR.name(), 1);
      game(out, 5, 1);
      hand(out, 0, 0, 0, even, code);
    }));
    CountedHands read = CountFile.read(written);
    Assert.assertEquals(read.hands(), 1);
    Assert.assertEquals(read.code(0), code);
  }

  /**
   * ヘッダーの Φ（モード・打ち切り）は往復で変わらない。打ち切りの無いモードの NaN もそのまま読める。手で書いたヘッダーで、Φ の
   * フィールドが探索の温度の後・半荘の数の前に並ぶことも確かめる。
   */
  @Test
  public void theHeaderCarriesThePhiStamp() throws IOException {
    Path directory = Files.createTempDirectory("count-stamp");
    CountedHands played = RulesFixtures.play(FOUR, 3, 1, 89).counts();
    List<PhiStamp> stamps =
        List.of(
            GrpFixtures.STAMP,
            new PhiStamp("LATE", 1e-5),
            new PhiStamp("LATE", 1e-6),
            new PhiStamp("LATE", 0));
    for (int at = 0; at < stamps.size(); at++) {
      PhiStamp stamp = stamps.get(at);
      CountedHands.Builder builder = CountedHands.builder(FOUR, 7, 0.5, stamp);
      for (int game = 0; game < played.games(); game++) {
        add(builder, played, game, played.gameId(game));
      }
      CountedHands counts = builder.build();
      Assert.assertEquals(counts.phi(), stamp);
      Path file = directory.resolve(at + ".bin");
      CountFile.write(file, counts);
      CountedHands read = CountFile.read(file);
      Assert.assertEquals(read.phi(), stamp);
      Assert.assertEquals(Double.isNaN(read.phi().cutoff()), Double.isNaN(stamp.cutoff()));
      Assert.assertEquals(read.exploreTemperature(), 0.5);
      assertSame(read, counts);
    }

    PhiStamp late = new PhiStamp("LATE", 1e-5);
    Path handmade = directory.resolve("handmade-late.bin");
    long code = RulesFixtures.allNoten(FOUR);
    Files.write(handmade, gzip(out -> {
      header(out, FOUR.name(), late.mode(), late.cutoff(), 1);
      game(out, 5, 1);
      hand(out, 0, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000}, code);
    }));
    Assert.assertEquals(CountFile.read(handmade).phi(), late);
  }

  /** 壊れた Φ のフィールド（空のモード・範囲の外の打ち切り）は、作るときも、ファイルから読むときも拒む。 */
  @Test
  public void brokenStampsAreRejected() throws IOException {
    Assert.assertThrows(NullPointerException.class, () -> new PhiStamp(null, Double.NaN));
    Assert.assertThrows(NullPointerException.class, () -> CountedHands.builder(FOUR, 1, 0, null));
    Path directory = Files.createTempDirectory("count-stamp-broken");
    long code = RulesFixtures.allNoten(FOUR);
    int[] even = {25_000, 25_000, 25_000, 25_000};
    String[] modes = {"", "LATE", "LATE", "LATE", "LATE"};
    double[] cutoffs = {Double.NaN, 1, -1e-9, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
    for (int at = 0; at < modes.length; at++) {
      String mode = modes[at];
      double cutoff = cutoffs[at];
      Assert.assertThrows(IllegalArgumentException.class, () -> new PhiStamp(mode, cutoff));
      assertRejected(directory, at + ".bin", gzip(out -> {
        header(out, FOUR.name(), mode, cutoff, 1);
        game(out, 5, 1);
        hand(out, 0, 0, 0, even, code);
      }));
    }
  }

  @Test
  public void buildersCheckWhatTheyCount() {
    CountedHands.Builder builder = CountedHands.builder(FOUR, 1, 0, GrpFixtures.STAMP);
    TableState start = new TableState(FOUR);
    long code = RulesFixtures.allNoten(FOUR);
    Assert.assertThrows(IllegalArgumentException.class, () -> CountedHands.builder(FOUR, 1, -1, GrpFixtures.STAMP));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> CountedHands.builder(FOUR, 1, Double.NaN, GrpFixtures.STAMP));
    Assert.assertThrows(IllegalArgumentException.class, () -> builder.add(1, new TableState(Variant.TENHOU_3P), code));
    Assert.assertThrows(IllegalArgumentException.class, () -> builder.add(1, start, -1L));
    TableState scenario = new TableState(FOUR);
    scenario.setScenario(0, 0, 0, new int[] {25_000, 25_000, 25_000, 24_000});
    Assert.assertThrows(IllegalArgumentException.class, () -> builder.add(1, scenario, code));
    builder.add(1, start, code).add(2, start, code).add(1, start, code);
    Assert.assertThrows(IllegalStateException.class, builder::build);
  }

  @Test
  public void mismatchesPointAtTheFirstBrokenHand() {
    CountedHands counts = RulesFixtures.play(FOUR, 30, 1, 83).counts();
    Assert.assertEquals(counts.firstMismatch(), -1);
    // 3 局以上ある半荘の 2 局目を、親でない席の 6 倍役満のツモ（誰かが飛ぶ）に替える
    int game = 0;
    while (counts.firstHand(game + 1) - counts.firstHand(game) < 3) {
      game++;
    }
    int target = counts.firstHand(game) + 1;
    CountedHands.Builder builder = CountedHands.builder(FOUR, 1, 0, GrpFixtures.STAMP);
    TableState start = new TableState(FOUR);
    long other = RulesFixtures.code(FOUR, 0, o -> o.setTsumo(BaseClass.YAKUMAN_6.base(), 1, RoundOutcome.NO_SEAT, 0));
    for (int at = 0; at < counts.games(); at++) {
      for (int hand = counts.firstHand(at); hand < counts.firstHand(at + 1); hand++) {
        counts.start(hand, start);
        builder.add(counts.gameId(at), start, hand == target ? other : counts.code(hand));
      }
    }
    Assert.assertEquals(builder.build().firstMismatch(), target);
  }

  private static void add(CountedHands.Builder builder, CountedHands played, int game, long gameId) {
    TableState start = new TableState(played.variant());
    for (int hand = played.firstHand(game); hand < played.firstHand(game + 1); hand++) {
      played.start(hand, start);
      builder.add(gameId, start, played.code(hand));
    }
  }

  private static void assertSame(CountedHands actual, CountedHands expected) {
    Assert.assertEquals(actual.variant(), expected.variant());
    Assert.assertEquals(actual.games(), expected.games());
    Assert.assertEquals(actual.hands(), expected.hands());
    int seats = expected.variant().seats();
    for (int game = 0; game <= expected.games(); game++) {
      Assert.assertEquals(actual.firstHand(game), expected.firstHand(game));
    }
    for (int game = 0; game < expected.games(); game++) {
      Assert.assertEquals(actual.gameId(game), expected.gameId(game));
    }
    for (int hand = 0; hand < expected.hands(); hand++) {
      Assert.assertEquals(actual.round(hand), expected.round(hand));
      Assert.assertEquals(actual.honba(hand), expected.honba(hand));
      Assert.assertEquals(actual.kyotaku(hand), expected.kyotaku(hand));
      Assert.assertEquals(actual.code(hand), expected.code(hand));
      for (int seat = 0; seat < seats; seat++) {
        Assert.assertEquals(actual.score(hand, seat), expected.score(hand, seat));
      }
    }
  }

  private static void assertRejected(Path directory, String name, byte[] bytes) throws IOException {
    Path file = directory.resolve(name);
    Files.write(file, bytes);
    Assert.assertThrows(IOException.class, () -> CountFile.read(file));
  }

  /** ファイルの中身を書く処理。IOException を投げてよい。 */
  @FunctionalInterface
  private interface Writing {
    void accept(DataOutputStream out) throws IOException;
  }

  private static byte[] gzip(Writing writing) throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (DataOutputStream out = new DataOutputStream(new GZIPOutputStream(bytes))) {
      writing.accept(out);
    }
    return bytes.toByteArray();
  }

  private static void header(DataOutputStream out, String variant, int games) throws IOException {
    header(out, variant, GrpFixtures.STAMP.mode(), GrpFixtures.STAMP.cutoff(), games);
  }

  private static void header(DataOutputStream out, String variant, String mode, double cutoff, int games)
      throws IOException {
    out.writeUTF(variant);
    out.writeInt(1);
    out.writeDouble(0);
    out.writeUTF(mode);
    out.writeDouble(cutoff);
    out.writeInt(games);
  }

  private static void game(DataOutputStream out, long gameId, int hands) throws IOException {
    out.writeLong(gameId);
    out.writeShort(hands);
  }

  private static void hand(DataOutputStream out, int round, int honba, int kyotaku, int[] scores, long code)
      throws IOException {
    out.writeByte(round);
    out.writeShort(honba);
    out.writeShort(kyotaku);
    for (int score : scores) {
      out.writeInt(score);
    }
    out.writeLong(code);
  }
}
