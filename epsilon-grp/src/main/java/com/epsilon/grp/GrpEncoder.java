package com.epsilon.grp;

import com.epsilon.mahjong.rules.TableState;
import java.util.Objects;

/** GRP の入力のフィールドを書く。GrpLayout のほかに GrpState を写す卓を持つので、書くときにメモリを確保しない。同時に使えるのは 1 スレッドだけ。 */
public final class GrpEncoder {

  private final GrpLayout layout;

  private final TableState table;

  private GrpEncoder(GrpLayout layout) {
    this.layout = layout;
    table = new TableState(layout.variant());
  }

  public static GrpEncoder of(GrpLayout layout) {
    return new GrpEncoder(Objects.requireNonNull(layout, "layout"));
  }

  public GrpLayout layout() {
    return layout;
  }

  /**
   * 卓を入力に書く。卓は TableState の決まり（RoundStart と同じ。点の和も含む）を満たすこと。
   *
   * @param out 書き先。先頭から入力の幅の分だけ書く
   */
  public void encode(GrpState state, float[] out) {
    layout.write(load(state), out);
  }

  /**
   * 半荘が続く卓を入力に書く。
   *
   * @param out 書き先。先頭から入力の幅の分だけ書く
   */
  public void encode(TableState table, float[] out) {
    layout.write(table, out);
  }

  /**
   * 渡した卓を作業用の卓に写す。ルールが違えば IllegalArgumentException を投げる。
   *
   * @return 写した卓。次の呼び出しまで有効
   */
  TableState load(GrpState state) {
    state.copyInto(table);
    return table;
  }
}
