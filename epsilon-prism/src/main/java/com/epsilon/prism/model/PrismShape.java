package com.epsilon.prism.model;

import com.epsilon.prism.input.FrontierField;

/**
 * 網の形。model.json に保存し、開くときに照らす。
 *
 * <p>注意の幅 A = H/2（頭 16 次元 × H/32 頭）と RMSNorm の ε = 1e-5·256/H は幅から決まる。ε を幅に反比例させるので、幅を広げても
 * RMSNorm の関数が保たれる（{@link PrismWiden}）。
 *
 * @param hidden 流れの幅 H（32 の倍数）
 * @param ffn FFN の幅 F（128 の倍数。背骨と候補層で共通）
 * @param gate 頭の隠れ幅 G（64 の倍数。B と門で共通）
 * @param depth 背骨の層の数 L（2 か 3。最後の層だけ問い合わせを局・席・牌に限る）
 * @param frontier 前線の腕なら true。候補層が遷移の入力に {@link FrontierField} の列を足して読むので、入力に前線の区画
 *     （{@link NetworkInputs#frontier()}）が要る
 */
public record PrismShape(int hidden, int ffn, int gate, int depth, boolean frontier) {

  public static final int HEAD_WIDTH = 16;

  /** 背骨の層の数の下限。 */
  public static final int MIN_DEPTH = 2;

  /** 背骨の層の数の上限。 */
  public static final int MAX_DEPTH = 3;

  /** 背骨の層の数の既定。 */
  public static final int DEFAULT_DEPTH = 2;

  /** ε を決める基準の幅。 */
  private static final double EPSILON_WIDTH = 256;

  /** 基準の幅での ε。 */
  private static final double BASE_EPSILON = 1e-5;

  /**
   * 形を確かめる。
   *
   * @throws IllegalArgumentException 幅が倍数の決まりを外れるか、深さが範囲の外のとき
   */
  public PrismShape {
    if (hidden <= 0 || hidden % (2 * HEAD_WIDTH) != 0) {
      throw new IllegalArgumentException("hidden must be a positive multiple of 32: " + hidden);
    }
    if (ffn <= 0 || ffn % 128 != 0) {
      throw new IllegalArgumentException("ffn must be a positive multiple of 128: " + ffn);
    }
    if (gate <= 0 || gate % 64 != 0) {
      throw new IllegalArgumentException("gate must be a positive multiple of 64: " + gate);
    }
    if (depth < MIN_DEPTH || depth > MAX_DEPTH) {
      throw new IllegalArgumentException("depth must be 2 or 3: " + depth);
    }
  }

  /**
   * 系譜の始めの形（H 256・F 512・G 256・深さは既定の 2・前線なし）を返す。
   *
   * @return 形
   */
  public static PrismShape standard() {
    return standard(DEFAULT_DEPTH);
  }

  /**
   * 系譜の始めの幅（H 256・F 512・G 256）・前線なしで、深さを選んだ形を返す。
   *
   * @param depth 背骨の層の数
   * @return 形
   */
  public static PrismShape standard(int depth) {
    return new PrismShape(256, 512, 256, depth, false);
  }

  /**
   * 注意の幅 A を返す。
   *
   * @return H/2
   */
  public int attention() {
    return hidden / 2;
  }

  /**
   * 頭の数を返す。
   *
   * @return H/32
   */
  public int heads() {
    return attention() / HEAD_WIDTH;
  }

  /**
   * RMSNorm の ε を返す。
   *
   * @return 1e-5·256/H
   */
  public double epsilon() {
    return BASE_EPSILON * EPSILON_WIDTH / hidden;
  }

  /**
   * 流れの幅だけを替えた形を返す。
   *
   * @param width 新しい H
   * @return 形
   */
  public PrismShape withHidden(int width) {
    return new PrismShape(width, ffn, gate, depth, frontier);
  }

  /**
   * FFN の幅だけを替えた形を返す。
   *
   * @param width 新しい F
   * @return 形
   */
  public PrismShape withFfn(int width) {
    return new PrismShape(hidden, width, gate, depth, frontier);
  }

  /**
   * 前線の腕の有無だけを替えた形を返す。
   *
   * @param arm 前線の腕なら true
   * @return 形
   */
  public PrismShape withFrontier(boolean arm) {
    return new PrismShape(hidden, ffn, gate, depth, arm);
  }
}
