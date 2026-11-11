package com.epsilon.reviewer.record;

import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.store.OwnerId;
import com.epsilon.reviewer.store.ResultId;
import com.epsilon.reviewer.store.ResultMeta;
import com.epsilon.reviewer.store.ResultStore;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;

/**
 * 牌譜を取り込む。バイト列を RecordLoader で読み、再生して確かめてから PreparedRecords に預ける。解析結果の文書を渡すと
 * （gzip で圧縮したものも）共通のリーダーが断るので、invalid_record になる。ファイル名は受け取らない。
 */
public final class RecordImporter {
  /** 牌譜のサイズの上限。ちょうど 16,777,216 バイトまでは受け取る。 */
  public static final int RECORD_LIMIT = 16 * 1024 * 1024;

  private final PreparedRecords records;
  private final TenhouFetcher tenhou;
  private final ResultStore store;

  public RecordImporter(PreparedRecords records, TenhouFetcher tenhou, ResultStore store) {
    this.records = records;
    this.tenhou = tenhou;
    this.store = store;
  }

  /** @throws ReviewerException record_too_large・invalid_record・too_many_records */
  public ImportedRecord importBytes(OwnerId owner, byte[] bytes) {
    if (bytes.length > RECORD_LIMIT) throw new ReviewerException(ErrorCode.RECORD_TOO_LARGE);
    return prepare(owner, bytes);
  }

  /**
   * デスクトップ版で、手元のファイルを取り込む。受け取るのは絶対パスで指定した通常のファイルで、16 MiB までに限る。
   *
   * @throws ReviewerException invalid_request・record_too_large・invalid_record・too_many_records
   */
  public ImportedRecord importFile(OwnerId owner, Path path) {
    if (!path.isAbsolute() || !Files.isRegularFile(path))
      throw new ReviewerException(
          ErrorCode.INVALID_REQUEST, "path must be the absolute path of a regular file.");
    byte[] bytes;
    try (InputStream in = Files.newInputStream(path)) {
      bytes = in.readNBytes(RECORD_LIMIT + 1);
    } catch (IOException failure) {
      throw new ReviewerException(ErrorCode.INVALID_REQUEST, "The file could not be read.");
    }
    return importBytes(owner, bytes);
  }

  /** @throws ReviewerException TenhouLogId と TenhouFetcher が投げるもの・invalid_record・too_many_records */
  public ImportedRecord importTenhou(OwnerId owner, String url) {
    return prepare(owner, tenhou.fetch(TenhouLogId.parse(url)));
  }

  /**
   * 保存した結果から、元の牌譜を読み込み直す。解析をやり直すときに使う。
   *
   * @throws ReviewerException result_not_found・too_many_records
   */
  public ImportedRecord importStored(OwnerId owner, ResultId id) {
    ResultMeta meta = store.find(owner, id);
    byte[] bytes;
    try {
      bytes = store.readRecord(meta);
    } catch (NoSuchFileException failure) {
      throw new ReviewerException(ErrorCode.RESULT_NOT_FOUND);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    return prepare(owner, bytes);
  }

  private ImportedRecord prepare(OwnerId owner, byte[] bytes) {
    LoadedRecord record;
    try {
      record = RecordLoader.load(bytes);
    } catch (IllegalArgumentException failure) {
      throw new ReviewerException(ErrorCode.INVALID_RECORD, failure.getMessage());
    }
    PreparedRecord prepared = records.add(owner, record, bytes);
    return new ImportedRecord(prepared.id().value(), record.summary());
  }
}
