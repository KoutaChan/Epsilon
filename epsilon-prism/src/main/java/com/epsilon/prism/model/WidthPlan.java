package com.epsilon.prism.model;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * パラメーターの全ての軸の役。網が登録のときに軸ごとに宣言し、{@link PrismWiden} がこれだけを見て広げる。名前の例外は持たない。
 *
 * <p>軸は役と塊の数を持つ。塊のある軸（例えば q・k・v を並べた [3A, H] の行）は塊ごとに広がり、古い塊 b の添字は新しい塊 b の先頭に
 * 写る。広げたときの新しい添字の値は役で決まる。
 *
 * <ul>
 *   <li>書く側（{@link Role#writes()}: 流れへの出力・頭の読み・素子の読み）の軸で新しい添字は 0。関数が変わらない。
 *   <li>読む側だけの新しい添字は、広い網の白紙の値（乱数か、0 で始める層なら 0）。1 歩目から勾配が届く。
 *   <li>RMSNorm の利得は古い成分を √(H/H′) 倍し、新しい成分を √(H/H′) にする。ε は形が H/H′ 倍にする。
 * </ul>
 */
public final class WidthPlan {

  /** 軸の役。 */
  public enum Role {
    /** 流れ（幅 H）に書く出力の軸。 */
    STREAM_OUT(true),
    /** 正規化した流れ（幅 H）を読む入力の軸。 */
    STREAM_IN(false),
    /** 頭の値を作る軸（幅 A の q・k・v、頭ごとの表の頭の軸）。 */
    HEAD_OUT(false),
    /** 頭の値を読んで流れに戻す入力の軸。 */
    HEAD_IN(true),
    /** 素子を作る軸（FFN の F、頭の G）。 */
    INNER_OUT(false),
    /** 素子を読む入力の軸。 */
    INNER_IN(true),
    /** 埋め込みの表の語彙の軸（広げない）。 */
    VOCAB(false),
    /** 入力の欄の軸（広げない）。 */
    IN(false),
    /** RMSNorm の利得（幅 H の 1 次元）。 */
    GAIN(false),
    /** 幅に依らない軸（広げない）。 */
    FIXED(false);

    private final boolean writes;

    Role(boolean writes) {
      this.writes = writes;
    }

    /**
     * 新しい添字を 0 にする軸なら true を返す。
     *
     * @return 書く側の軸なら true
     */
    public boolean writes() {
      return writes;
    }

    /**
     * 幅を広げても変わらない軸なら true を返す。
     *
     * @return 広げない軸なら true
     */
    public boolean fixed() {
      return this == VOCAB || this == IN || this == FIXED;
    }
  }

  /**
   * 1 つの軸の役。
   *
   * @param role 役
   * @param blocks 塊の数（1 以上）
   */
  public record Axis(Role role, int blocks) {

    /**
     * 役と塊を確かめる。
     *
     * @throws IllegalArgumentException 塊の数が 1 未満のとき
     */
    public Axis {
      Objects.requireNonNull(role, "role");
      if (blocks < 1) {
        throw new IllegalArgumentException("blocks must be positive: " + blocks);
      }
    }

    /**
     * 塊が 1 つの軸を返す。
     *
     * @param role 役
     * @return 軸
     */
    public static Axis of(Role role) {
      return new Axis(role, 1);
    }

    /**
     * 塊が blocks 個の軸を返す。
     *
     * @param role 役
     * @param blocks 塊の数
     * @return 軸
     */
    public static Axis of(Role role, int blocks) {
      return new Axis(role, blocks);
    }
  }

  private final Map<String, List<Axis>> axes;

  /**
   * パラメーターの名前（網の登録の順）から軸の役への表を作る。
   *
   * @param axes 名前から軸の並びへの表
   */
  WidthPlan(Map<String, List<Axis>> axes) {
    this.axes = Collections.unmodifiableMap(new LinkedHashMap<>(axes));
  }

  /**
   * パラメーターの名前を登録の順に返す。
   *
   * @return 名前
   */
  public Set<String> parameters() {
    return axes.keySet();
  }

  /**
   * パラメーターの軸の役を返す。
   *
   * @param parameter パラメーターの名前
   * @return 軸の並び
   * @throws IllegalArgumentException 名前がないとき
   */
  public List<Axis> axes(String parameter) {
    List<Axis> result = axes.get(parameter);
    if (result == null) {
      throw new IllegalArgumentException("unknown parameter: " + parameter);
    }
    return result;
  }
}
