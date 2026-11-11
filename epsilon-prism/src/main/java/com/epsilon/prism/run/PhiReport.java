package com.epsilon.prism.run;

import com.epsilon.prism.grp.PhiValue;
import java.util.Objects;

/**
 * Φ の段の報告。work の PHI.json に書き、このファイルが段の完了マーカーになる。収集の段は、ここに書いた値の Φ を使って集める。
 *
 * @param value 反復で使う Φ の値（GRP の項目、求めたモードと実際に使ったモード、打ち切りの値）
 * @param waitSeconds GRP のバックグラウンドの作業が前の反復の区切りを終えるまで待った時間（秒）
 */
public record PhiReport(PhiValue value, double waitSeconds) {

  public PhiReport {
    Objects.requireNonNull(value, "value");
  }
}
