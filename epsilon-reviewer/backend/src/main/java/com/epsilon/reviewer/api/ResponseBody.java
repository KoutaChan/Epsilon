package com.epsilon.reviewer.api;

/** 応答の本文（protocol/commands.ts の ResponseBody）。 */
public enum ResponseBody {
  /** JSON（HTTP は 200）。 */
  JSON,
  /** 本文なし（HTTP は 204）。 */
  EMPTY,
  /** 保存した gzip の解析結果の文書（HTTP は Content-Encoding: gzip、stdio は枠の本文）。 */
  DOCUMENT,
  /** HTTP のダウンロード（application/gzip の添付）。 */
  ATTACHMENT
}
