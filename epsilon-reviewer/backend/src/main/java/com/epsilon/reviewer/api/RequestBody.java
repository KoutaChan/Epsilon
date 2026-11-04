package com.epsilon.reviewer.api;

/** 要求の本文（protocol/commands.ts の RequestBody）。 */
public enum RequestBody {
  /** 道の欄だけ。 */
  NONE,
  /** 道の欄を除いた欄の JSON。 */
  JSON,
  /** 牌譜のバイト列。 */
  BINARY
}
