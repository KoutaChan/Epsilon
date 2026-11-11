package com.epsilon.reviewer;

import com.epsilon.reviewer.store.ResultId;
import com.epsilon.reviewer.store.StoreLayout;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.UnaryOperator;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/** 保存先の結果のフォルダーを直接読み書きする、テスト用のヘルパー。保存先を開く前に使う。 */
public final class StoredResults {
  private StoredResults() {}

  /** 結果を新しい id でコピーする。牌譜と文書は渡した関数で書き換えてから書き、meta.json の id も新しい id に置き換える。 */
  public static ResultId copy(
      Path data, ResultId source, UnaryOperator<byte[]> record, UnaryOperator<byte[]> document) {
    ResultId id = ResultId.random();
    Path from = folder(data, source);
    Path to = folder(data, id);
    String meta = new String(read(from, StoreLayout.META), StandardCharsets.UTF_8);
    write(to.resolve(StoreLayout.RECORD.fileName()), record.apply(read(from, StoreLayout.RECORD)));
    write(
        to.resolve(StoreLayout.DOCUMENT.fileName()),
        document.apply(read(from, StoreLayout.DOCUMENT)));
    write(
        to.resolve(StoreLayout.META.fileName()),
        meta.replace(source.value(), id.value()).getBytes(StandardCharsets.UTF_8));
    return id;
  }

  public static byte[] read(Path data, ResultId id, StoreLayout file) {
    return read(folder(data, id), file);
  }

  public static byte[] gunzip(byte[] gzip) {
    try (var input = new GZIPInputStream(new ByteArrayInputStream(gzip))) {
      return input.readAllBytes();
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  public static byte[] gzip(byte[] bytes) {
    var out = new ByteArrayOutputStream();
    try (var gzip = new GZIPOutputStream(out)) {
      gzip.write(bytes);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    return out.toByteArray();
  }

  private static Path folder(Path data, ResultId id) {
    return data.resolve(StoreLayout.RESULTS.fileName()).resolve(id.value());
  }

  private static byte[] read(Path folder, StoreLayout file) {
    try {
      return Files.readAllBytes(folder.resolve(file.fileName()));
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  private static void write(Path file, byte[] bytes) {
    try {
      Files.createDirectories(file.getParent());
      Files.write(file, bytes);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }
}
