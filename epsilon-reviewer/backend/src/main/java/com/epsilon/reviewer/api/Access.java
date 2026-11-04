package com.epsilon.reviewer.api;

/** 誰が操作を呼べるか（protocol/commands.ts の Access）。 */
public enum Access {
  /** Cookie を読まず発行もしない公開の道。 */
  PUBLIC,
  /** Cookie の持ち主（無ければ発行する）。 */
  OWNER,
  /** 共有リンク（id を知る誰でも、Cookie を読まない）。 */
  SHARE,
  /** デスクトップの Java の子プロセス（stdio だけ）。 */
  LOCAL
}
