package com.epsilon.reviewer.store;

/** 保存先に置くファイルとフォルダーの名前。Web のサーバとデスクトップで同じ配置にする。 */
public enum StoreLayout {
  /** 動いている間、OS のロックを取っておくファイル。 */
  LOCK("reviewer.lock"),
  /** 保存した結果のフォルダー（results/&lt;resultId&gt;/）。 */
  RESULTS("results"),
  /** 保存している途中のフォルダー（staging/&lt;resultId&gt;/）。起動時に消す。 */
  STAGING("staging"),
  /** 削除した結果のフォルダー（trash/&lt;ランダムな UUID&gt;/）。起動時と {@link ResultStore#sweep} で消す。 */
  TRASH("trash"),
  /** 解析結果の文書（レベル 9 の gzip）。 */
  DOCUMENT("result.json.gz"),
  /** 元の牌譜（レベル 9 の gzip）。受け取ったときに gzip だったものはそのまま置く。 */
  RECORD("record.gz"),
  /** インデックスの 1 件（ResultMeta）。最後に書く。 */
  META("meta.json");

  private final String fileName;

  StoreLayout(String fileName) {
    this.fileName = fileName;
  }

  public String fileName() {
    return fileName;
  }
}
