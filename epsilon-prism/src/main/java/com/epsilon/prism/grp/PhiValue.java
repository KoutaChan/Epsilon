package com.epsilon.prism.grp;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.grp.train.PhiStamp;
import java.util.Objects;

/**
 * 反復で使う Φ。GRP の項目（phi.jsonl の行か、固定の出どころの項目）に、モードの行を当てたもの。LATE のモードを求められても、
 * 項目が局の結果のネットワークを持たなければ GRP のモードで求める。
 *
 * @param entry GRP の項目
 * @param requested モードの行が求めたモード。効いている行が無ければ GRP
 * @param mode 使ったモード
 * @param cutoff LATE のモードの数え上げの打ち切りの値。GRP のモードでは無い
 */
public record PhiValue(
    PhiEntry entry,
    PhiMode requested,
    PhiMode mode,
    @Key(doc = "LATE のモードの数え上げの打ち切りの値。GRP のモードでは無い", presence = Presence.ABSENT) Double cutoff) {

  public PhiValue {
    Objects.requireNonNull(entry, "entry");
    Objects.requireNonNull(requested, "requested");
    Objects.requireNonNull(mode, "mode");
    if (mode == PhiMode.LATE && (requested != PhiMode.LATE || !entry.hasOutcomes() || cutoff == null)) {
      throw new IllegalArgumentException(
          "The LATE mode needs a LATE line, an outcome network and a cutoff: " + entry.id());
    }
    if (mode == PhiMode.GRP && cutoff != null) {
      throw new IllegalArgumentException("The GRP mode takes no cutoff: " + entry.id());
    }
  }

  /** 効いているモードの行が無い反復の値。GRP のモードになる。 */
  public static PhiValue grp(PhiEntry entry) {
    return of(entry, PhiMode.GRP, null);
  }

  /** 項目にモードの行を当てた値。 */
  public static PhiValue of(PhiEntry entry, ModeLine line) {
    return of(entry, line.mode(), line.cutoff());
  }

  /** 数のファイルのヘッダーに書く印。使ったモードと、打ち切りの値（GRP のモードでは NaN）を持つ。 */
  public PhiStamp stamp() {
    return new PhiStamp(mode.name(), cutoff == null ? Double.NaN : cutoff);
  }

  /**
   * どの局末でも同じ Φ を出すなら true。項目の id と中身（{@link PhiEntry#sameValue}）、使ったモード、打ち切りの値を比べ、求めたモードは
   * 見ない。固定の GRP の項目は id がいつも同じなので、ファイルを替えたことは、項目のパスと sha で見分ける。
   */
  public boolean sameValue(PhiValue other) {
    return entry.id().equals(other.entry.id())
        && entry.sameValue(other.entry)
        && mode == other.mode
        && Objects.equals(cutoff, other.cutoff);
  }

  /**
   * 項目に、求めたモードと打ち切りの値を当てた値。打ち切りの値を持つのは LATE のモードの値だけ。
   *
   * @param cutoff LATE の打ち切りの値。GRP のモードになるときは読まない
   * @throws IllegalArgumentException LATE のモードになるのに、打ち切りの値が無いとき
   */
  static PhiValue of(PhiEntry entry, PhiMode requested, Double cutoff) {
    PhiMode mode = requested == PhiMode.LATE && entry.hasOutcomes() ? PhiMode.LATE : PhiMode.GRP;
    return new PhiValue(entry, requested, mode, mode == PhiMode.LATE ? cutoff : null);
  }
}
