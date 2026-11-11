package com.epsilon.reviewer.store;

import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;

/** 保存している途中の 1 件（staging/&lt;resultId&gt;/）。文書と牌譜を書いてから、{@link ResultStore#commit} で確定する。 */
public final class ResultDraft implements Closeable {
  private final ResultId id;
  private final Path directory;
  private SyncedGzipFile document;
  private boolean recorded;
  private boolean committed;

  ResultDraft(ResultId id, Path directory) {
    this.id = id;
    this.directory = directory;
  }

  public ResultId id() {
    return id;
  }

  /** 文書を書くストリームを開く。開けるのは 1 回だけ。gzip（レベル 9）で書き、閉じると gzip の末尾を書いてディスクに同期する。 */
  public OutputStream document() throws IOException {
    if (document != null) throw new IllegalStateException("The document is already open.");
    document = SyncedGzipFile.create(directory.resolve(StoreLayout.DOCUMENT.fileName()));
    return document;
  }

  /** 元の牌譜を書く。gzip（先頭が 1f 8b）ならそのまま書き、それ以外はレベル 9 の gzip で圧縮して書く。 */
  public void record(byte[] original) throws IOException {
    Path path = directory.resolve(StoreLayout.RECORD.fileName());
    if (original.length >= 2 && (original[0] & 0xff) == 0x1f && (original[1] & 0xff) == 0x8b)
      ResultStore.writeFile(path, original);
    else
      try (var out = SyncedGzipFile.create(path)) {
        out.write(original);
      }
    recorded = true;
  }

  /** 確定の前に文書を閉じ、この 1 件のフォルダーを返す。文書と牌譜をまだ書いていなければ IllegalStateException を投げる。 */
  Path seal() throws IOException {
    if (document == null || !recorded)
      throw new IllegalStateException("The document and the record must be written first.");
    document.close();
    return directory;
  }

  void committed() {
    committed = true;
  }

  /** 確定していなければ、書いたものを捨てる。 */
  @Override
  public void close() {
    if (committed) return;
    if (document != null)
      try {
        document.close();
      } catch (IOException ignored) {
        // 捨てる文書の末尾を書けなくても、フォルダーごと消す。
      }
    ResultStore.discard(directory);
  }
}
