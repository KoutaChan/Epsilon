package com.epsilon.reviewer.api;

/** 応答の本文（protocol/commands.ts の ResponseBody）。 */
public enum ResponseBody {
  /** JSON（HTTP は 200）。 */
  JSON,
  /** 本文なし（HTTP は 204）。 */
  EMPTY,
  /** 解析結果の文書を、保存した gzip のまま返す。HTTP では Content-Encoding: gzip を付け、stdio ではフレームの本文にする。 */
  DOCUMENT,
  /** HTTP のダウンロード（application/gzip の添付）。 */
  ATTACHMENT
}
