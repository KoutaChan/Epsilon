package com.epsilon.reviewer.api;

/** 要求の本文（protocol/commands.ts の RequestBody）。 */
public enum RequestBody {
  /** 本文なし。要求のフィールドは HTTP のパスにあるものだけ。 */
  NONE,
  /** パスにあるフィールドを除いた、残りのフィールドの JSON。 */
  JSON,
  /** 牌譜のバイト列。 */
  BINARY
}
