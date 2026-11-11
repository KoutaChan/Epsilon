package com.epsilon.reviewer.api;

/** 誰が操作を呼べるか（protocol/commands.ts の Access）。 */
public enum Access {
  /** Cookie を読まず、発行もしない公開のパス。 */
  PUBLIC,
  /** Cookie で見分けたオーナー（Cookie が無ければ発行する）。 */
  OWNER,
  /** 共有リンク。id を知っていれば誰でも呼べる。Cookie は読まない。 */
  SHARE,
  /** デスクトップアプリが、Java の子プロセスの stdio を通して呼ぶ。stdio からしか呼べない。 */
  LOCAL
}
