package com.epsilon.reviewer.store;

/** 保存先の名前（Web のサーバとデスクトップで同じ配置）。 */
public enum StoreLayout {
  /** 動いている間 OS の鍵を持つファイル。 */
  LOCK("reviewer.lock"),
  /** 保存した結果のフォルダー（results/&lt;resultId&gt;/）。 */
  RESULTS("results"),
  /** 保存の途中のフォルダー（staging/&lt;resultId&gt;/。起動時に消す）。 */
  STAGING("staging"),
  /** 消した結果のフォルダー（trash/&lt;乱数の UUID&gt;/。起動時と掃除で消す）。 */
  TRASH("trash"),
  /** 解析結果の文書（gzip 9）。 */
  DOCUMENT("result.json.gz"),
  /** 元の牌譜（gzip 9。受け取った時に gzip ならそのまま）。 */
  RECORD("record.gz"),
  /** 索引の 1 件（ResultMeta）。最後に書く。 */
  META("meta.json");

  private final String fileName;

  StoreLayout(String fileName) {
    this.fileName = fileName;
  }

  public String fileName() {
    return fileName;
  }
}
