package com.epsilon.classic.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import java.util.Arrays;
import java.util.Objects;

/**
 * 行の大きさ（合法手の軸と遷移の軸）の区分。区分で変わるのはパディングの量だけで、重みには関わらない。合法手の軸は
 * {1, 2, 4, 8, 10, 12, 14, 16, 32, 40}、遷移の軸は {1, 4, 8, 12, 16} のどれか。どの合法手にも遷移が 1 つ以上ある。チー・ポンは鳴いた後の
 * 打牌ごとに 1 つ、ほかの手は 1 つ。
 *
 * <p>区分を選ぶメソッドは前もって作った区分を返すので、メモリを確保しない。区分の通し番号は、合法手の軸の位置 × 遷移の軸の区分の数 ＋
 * 遷移の軸の位置。
 */
public final class Buckets {

  /** 合法手の軸の区分（昇順）。 */
  private static final int[] ACTIONS = {1, 2, 4, 8, 10, 12, 14, 16, 32, ClassicSchema.MAX_LEGAL};

  /** 遷移の軸の区分（昇順）。 */
  private static final int[] TRANSITIONS = {1, 4, 8, 12, ClassicSchema.MAX_TRANSITIONS};

  /** 推論で、遷移の軸が 1 の行をまとめる合法手の軸の大きさ。 */
  private static final int SHARED_ACTIONS = 16;

  /** [通し番号] → 区分。 */
  private static final Bucket[] ALL = all();

  private Buckets() {}

  /**
   * 行の大きさの区分。
   *
   * @param actions 合法手の軸の大きさ
   * @param transitions 合法手 1 つあたりの遷移の軸の大きさ
   */
  public record Bucket(int actions, int transitions) {

    /** どちらかが区分の値でなければ IllegalArgumentException を投げる。 */
    public Bucket {
      if (ACTIONS[position(ACTIONS, actions)] != actions) {
        throw new IllegalArgumentException("Not a legal-action bucket: " + actions);
      }
      if (TRANSITIONS[position(TRANSITIONS, transitions)] != transitions) {
        throw new IllegalArgumentException("Not a transition bucket: " + transitions);
      }
    }

    /**
     * 行がこの区分に収まるなら true を返す。
     *
     * @param count 合法手の数
     * @param maxTransitions 合法手 1 つあたりの遷移の数の最大値
     */
    public boolean fits(int count, int maxTransitions) {
      return count <= actions && maxTransitions <= transitions;
    }
  }

  /**
   * 判断の行が収まる最小の区分を返す。学習の行に使う。
   *
   * @param count 合法手の数
   * @param maxTransitions 合法手 1 つあたりの遷移の数の最大値
   * @throws IllegalArgumentException どちらかが 1 未満か、上限を超えるとき
   */
  public static Bucket required(int count, int maxTransitions) {
    return ALL[ordinal(position(ACTIONS, count), position(TRANSITIONS, maxTransitions))];
  }

  /**
   * 推論のバッチで形をそろえやすい、粗い区分を返す。遷移の軸は、どの合法手も遷移が 1 つなら 1、そうでなければ 16。合法手の軸は、
   * 合法手が 1 つなら 1、16 を超えれば 40。それ以外のときは、遷移の軸が 1 なら 16 にまとめ、16 なら 4・8・16 のうち収まる最小のもの。
   *
   * @param count 合法手の数
   * @param maxTransitions 合法手 1 つあたりの遷移の数の最大値
   * @throws IllegalArgumentException どちらかが 1 未満か、上限を超えるとき
   */
  public static Bucket inference(int count, int maxTransitions) {
    required(count, maxTransitions);
    int transitions = maxTransitions == 1 ? 1 : ClassicSchema.MAX_TRANSITIONS;
    int actions;
    if (count == 1) {
      actions = 1;
    } else if (count > SHARED_ACTIONS) {
      actions = ClassicSchema.MAX_LEGAL;
    } else if (transitions == 1) {
      actions = SHARED_ACTIONS;
    } else {
      actions = count <= 4 ? 4 : count <= 8 ? 8 : SHARED_ACTIONS;
    }
    return ALL[ordinal(position(ACTIONS, actions), position(TRANSITIONS, transitions))];
  }

  /** 判断の行が収まる最小の区分を返す。 */
  public static Bucket required(Observation o) {
    return required(o.legalActions(), maxTransitions(o));
  }

  /** 推論で使う、判断の行の区分を返す。 */
  public static Bucket inference(Observation o) {
    return inference(o.legalActions(), maxTransitions(o));
  }

  public static int count() {
    return ALL.length;
  }

  /** 区分の通し番号（0 以上 {@link #count} 未満）。 */
  public static int ordinal(Bucket bucket) {
    return ordinal(position(ACTIONS, bucket.actions()), position(TRANSITIONS, bucket.transitions()));
  }

  /**
   * 通し番号から区分を引く。
   *
   * @throws IndexOutOfBoundsException 通し番号が範囲の外のとき
   */
  public static Bucket of(int ordinal) {
    return ALL[Objects.checkIndex(ordinal, ALL.length)];
  }

  /**
   * 合法手の遷移の数を返す。チー・ポンは鳴いた後に打てる牌の数、ほかの手は 1。
   *
   * @param action 合法手の何番目か
   */
  public static int transitions(Observation o, int action) {
    Action.Type type = o.action(action).type();
    return type == Action.Type.CHI || type == Action.Type.PON ? o.callDiscardsOf(action) : 1;
  }

  /** 判断の合法手のうち、遷移がいちばん多い手の遷移の数を返す。 */
  public static int maxTransitions(Observation o) {
    int max = 1;
    for (int action = 0; action < o.legalActions(); action++) {
      max = Math.max(max, transitions(o, action));
    }
    return max;
  }

  /** フィンガープリントに入れる、区分の並びの記述。 */
  static String describe() {
    return "actions=" + Arrays.toString(ACTIONS) + ";transitions=" + Arrays.toString(TRANSITIONS);
  }

  private static int ordinal(int actions, int transitions) {
    return actions * TRANSITIONS.length + transitions;
  }

  /**
   * 区分の並びのうち、必要な大きさ以上で最小の値の位置を返す。そのような値が無いか、必要な大きさが 1 未満なら
   * IllegalArgumentException を投げる。
   */
  private static int position(int[] buckets, int required) {
    if (required >= 1) {
      for (int index = 0; index < buckets.length; index++) {
        if (required <= buckets[index]) {
          return index;
        }
      }
    }
    throw new IllegalArgumentException("No bucket holds " + required);
  }

  private static Bucket[] all() {
    Bucket[] all = new Bucket[ACTIONS.length * TRANSITIONS.length];
    for (int actions = 0; actions < ACTIONS.length; actions++) {
      for (int transitions = 0; transitions < TRANSITIONS.length; transitions++) {
        all[ordinal(actions, transitions)] = new Bucket(ACTIONS[actions], TRANSITIONS[transitions]);
      }
    }
    return all;
  }
}
