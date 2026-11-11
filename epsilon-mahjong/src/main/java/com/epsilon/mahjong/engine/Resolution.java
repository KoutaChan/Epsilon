package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * 直前に選んだ手で局を進めた結果。席ごとに、選んだ手が行われたかと、選び替えで結果が変わりえたかを持つ。並びは、そのとき
 * 答えた {@link Decisions} と同じ。エンジンが持ち、進めるたびに書き直す。局を次に進める呼び出しまで有効。
 */
public final class Resolution {

  public enum Role {
    /** 他家の選んだ手はそのままで自分だけ選び替えると、結果が変わりえた。手番と鳴いた後の打牌では、合法手が 2 つ以上ならいつもこれ。 */
    CAUSAL,
    /** 合法手が 1 つだけで、それが行われた。 */
    FORCED,
    /** 他家のロンか優先する鳴きがあり、どの合法手を選んでも応答の結果は変わらなかった。 */
    PREEMPTED
  }

  private static final Role[] ROLES = Role.values();

  private final Variant variant;
  private final long[] ids;
  private final boolean[] executed;
  private final byte[] roles;
  private int size;

  Resolution(Variant variant) {
    this.variant = variant;
    int slots = variant.seats() - 1;
    ids = new long[slots];
    executed = new boolean[slots];
    roles = new byte[slots];
  }

  public int size() {
    return size;
  }

  /** 判断 ID。 */
  public long id(int slot) {
    return ids[Objects.checkIndex(slot, size)];
  }

  /** 選んだ手が行われたなら true。鳴きは、他家のロンか優先する鳴きがあると行われない。 */
  public boolean executed(int slot) {
    return executed[Objects.checkIndex(slot, size)];
  }

  public Role role(int slot) {
    return ROLES[roles[Objects.checkIndex(slot, size)]];
  }

  void clear() {
    size = 0;
  }

  void add(long id, boolean executedChoice, Role role) {
    if (size == ids.length) {
      throw new IllegalStateException("At most " + ids.length + " resolved decisions");
    }
    ids[size] = id;
    executed[size] = executedChoice;
    roles[size] = (byte) role.ordinal();
    size++;
  }

  void copyFrom(Resolution other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    System.arraycopy(other.ids, 0, ids, 0, ids.length);
    System.arraycopy(other.executed, 0, executed, 0, executed.length);
    System.arraycopy(other.roles, 0, roles, 0, roles.length);
    size = other.size;
  }
}
