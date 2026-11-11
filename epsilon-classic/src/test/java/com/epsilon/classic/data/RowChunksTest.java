package com.epsilon.classic.data;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 行のディレクトリの読み込みを確かめる。行のファイルを名前の順に読み、行と半荘の通し番号はファイルの順に続く（行の無いファイルを
 * 挟んでも同じ）。名前の形が違うファイルは読まず、行のファイルが 1 つも無ければ拒む。行のファイルのヘッダーにある観測の long の総数が
 * 中身と合わなければ拒む。
 */
public class RowChunksTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @Test
  public void numbersContinueAcrossTheFilesInNameOrder() throws Exception {
    ClassicRows first = rows(1, 2);
    ClassicRows empty = ClassicRows.builder(VARIANT).build();
    ClassicRows last = rows(3);
    Path directory = Files.createTempDirectory("row-chunks");
    try {
      // 名前の順とは違う順に書き、名前の形が違うファイルも置く
      last.write(RowChunks.file(directory, 2));
      empty.write(RowChunks.file(directory, 1));
      first.write(RowChunks.file(directory, 0));
      Files.writeString(directory.resolve("notes.txt"), "not rows");
      Files.writeString(directory.resolve("rows-00003.bin.tmp"), "not rows");

      RowChunks chunks = RowChunks.read(directory);
      Assert.assertEquals(chunks.chunks(), 3);
      Assert.assertEquals(chunks.variant(), VARIANT);
      Assert.assertEquals(chunks.size(), first.size() + last.size());
      List<ClassicRows.Game> games = new ArrayList<>(first.games());
      games.addAll(last.games());
      Assert.assertEquals(chunks.games(), games);

      Observation expected = new Observation(VARIANT);
      Observation actual = new Observation(VARIANT);
      for (int row = 0; row < chunks.size(); row++) {
        ClassicRows part = row < first.size() ? first : last;
        int local = row < first.size() ? row : row - first.size();
        int gameOffset = row < first.size() ? 0 : first.games().size();
        Assert.assertEquals(chunks.gameIndex(row), gameOffset + part.gameIndex(local), "row " + row);
        Assert.assertEquals(chunks.game(row), part.game(local), "row " + row);
        Assert.assertEquals(chunks.round(row), part.round(local), "row " + row);
        Assert.assertEquals(chunks.chosen(row), part.chosen(local), "row " + row);
        Assert.assertEquals(chunks.validation(row), part.validation(local), "row " + row);
        Observation o = chunks.bind(row, actual);
        Observation p = part.bind(local, expected);
        Assert.assertEquals(o.observer(), p.observer(), "row " + row);
        Assert.assertEquals(o.decisionKind(), p.decisionKind(), "row " + row);
        Assert.assertEquals(o.legalActions(), p.legalActions(), "row " + row);
      }
      Assert.assertThrows(IndexOutOfBoundsException.class, () -> chunks.chosen(chunks.size()));
      Assert.assertThrows(IndexOutOfBoundsException.class, () -> chunks.chosen(-1));
    } finally {
      delete(directory);
    }
  }

  @Test
  public void aDirectoryWithoutRowFilesIsRejected() throws Exception {
    Path directory = Files.createTempDirectory("row-chunks-none");
    try {
      Files.writeString(directory.resolve("rows.bin.txt"), "not rows");
      Assert.assertThrows(IOException.class, () -> RowChunks.read(directory));
    } finally {
      delete(directory);
    }
  }

  @Test
  public void aHeaderLongCountThatDisagreesWithTheRowsIsRejected() throws Exception {
    ClassicRows rows = rows(1);
    Path file = Files.createTempFile("classic-rows", ".bin");
    try {
      rows.write(file);
      Assert.assertEquals(ClassicRows.read(file).longs(), rows.longs());
      byte[] raw = gunzip(Files.readAllBytes(file));
      // ヘッダーの「i32 行の数、i32 観測の long の総数」を探し、総数を 1 増やす
      byte[] header = ByteBuffer.allocate(8).putInt(rows.size()).putInt(rows.longs()).array();
      int at = indexOf(raw, header);
      Assert.assertTrue(at > 0, "header not found");
      ByteBuffer.wrap(raw).putInt(at + 4, rows.longs() + 1);
      Files.write(file, gzip(raw));
      Assert.assertThrows(IOException.class, () -> ClassicRows.read(file));
    } finally {
      Files.deleteIfExists(file);
    }
  }

  /** シードごとの半荘を、ルールベースの打ち手で打って作った行。 */
  private static ClassicRows rows(long... seeds) {
    ClassicRows.Builder builder = ClassicRows.builder(VARIANT);
    LogRows logs = new LogRows(VARIANT, 0.1);
    for (long seed : seeds) {
      logs.add(RuleGames.play(VARIANT, seed, "rule-" + seed), builder);
    }
    return builder.build();
  }

  private static int indexOf(byte[] haystack, byte[] needle) {
    for (int at = 0; at + needle.length <= haystack.length; at++) {
      boolean found = true;
      for (int index = 0; index < needle.length && found; index++) {
        found = haystack[at + index] == needle[index];
      }
      if (found) {
        return at;
      }
    }
    return -1;
  }

  private static byte[] gunzip(byte[] bytes) throws IOException {
    try (InputStream in = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
      return in.readAllBytes();
    }
  }

  private static byte[] gzip(byte[] bytes) throws IOException {
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    try (OutputStream out = new GZIPOutputStream(buffer)) {
      out.write(bytes);
    }
    return buffer.toByteArray();
  }

  private static void delete(Path root) throws IOException {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}
