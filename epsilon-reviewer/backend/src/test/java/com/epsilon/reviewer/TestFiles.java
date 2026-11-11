package com.epsilon.reviewer;

import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.rules.Variant;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.zip.GZIPOutputStream;

/** テストで使う一時フォルダーと牌譜を扱うヘルパー。 */
public final class TestFiles {
  private TestFiles() {}

  public static Path directory(String prefix) {
    try {
      return Files.createTempDirectory(prefix);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /** フォルダーを中身ごと消す（無ければ何もしない）。 */
  public static void delete(Path directory) {
    if (!Files.exists(directory)) return;
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /**
   * エンジンで打った半荘（epsilon-mahjong のテスト用の jar の {@code RecordedGames}）を天鳳の XML に書き、gzip にした牌譜のバイト列。
   * 牌譜 id は played-種、打ち手の名前は空。
   *
   * @param seed 半荘を打つ乱数の種
   */
  public static byte[] played(Variant variant, long seed) {
    String xml = TenhouXml.write(RecordedGames.play(variant, seed, "played-" + seed));
    var out = new ByteArrayOutputStream();
    try (var gzip = new GZIPOutputStream(out)) {
      gzip.write(xml.getBytes(StandardCharsets.UTF_8));
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    return out.toByteArray();
  }
}
