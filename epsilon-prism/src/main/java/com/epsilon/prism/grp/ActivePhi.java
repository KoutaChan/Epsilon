package com.epsilon.prism.grp;

import com.epsilon.grp.Phi;

/**
 * 反復 t の収集で使う Φ。
 *
 * @param value 使う Φ の値。報告の Φ の段に書き、その GRP の項目はコホートの記録の先頭にも書く
 * @param phi 局末の値を求める Φ
 */
public record ActivePhi(PhiValue value, Phi phi) {

  /** 使う GRP の項目。 */
  public PhiEntry entry() {
    return value.entry();
  }
}
