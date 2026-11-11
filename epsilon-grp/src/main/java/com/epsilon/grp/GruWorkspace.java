package com.epsilon.grp;

import com.epsilon.mahjong.util.SeedMixer;

/**
 * GruSinkhorn の作業領域。局末の行、GRU の状態とゲート、出力ヘッドと Sinkhorn の配列、GRU の状態のメモを持つ。モデルごとに作り、
 * 作るときに渡したモデルでしか使えない。ほかのモデルに渡すと、形が同じでも IllegalArgumentException を投げる。同時に使えるのは
 * 1 スレッドだけ。
 *
 * <p>状態のメモは、RoundHistory のバージョンで引くダイレクトマップ方式。バージョンが一致したときだけ使い、入れるときにスロットが
 * 埋まっていれば上書きする。メモに入れるのは、このモデルで履歴の全ての局を読んだ後の各層の状態。状態はモデルと履歴だけで決まる
 * ので、メモに当たっても外れても値はビット単位で同じになる。
 */
public final class GruWorkspace {

  private static final int STATE_SLOTS = 1 << 8;

  final GruSinkhorn model;

  /** 局末の卓の行（RoundField の並び）。 */
  final int[] row;

  final float[] input;

  /** 添字は [層][ユニット]。 */
  final float[][] state;

  final float[] inputGates;
  final float[] stateGates;
  final float[] projected;
  final double[] logits;

  /** 最後に求めた周辺確率。並びは [絶対席 × S ＋ 順位]。 */
  final double[] marginals;

  final double[] rowMeans;
  final double[] columnMeans;

  /**
   * スロットごとの履歴のバージョン。空のスロットは 0 で、これは空の履歴のバージョンと同じ。GruSinkhorn は空の履歴を読まないので、
   * 空のスロットに当たることはない。
   */
  private final long[] stamps;

  /** スロットごとの状態。並びは [スロット × 層 × ユニット]。 */
  private final float[] states;

  /** 1 つのスロットの値の数（層 × ユニット）。 */
  private final int stride;

  private final int mask;
  private long stateHits;
  private long stateMisses;

  private GruWorkspace(GruSinkhorn model) {
    this.model = model;
    GruShape shape = model.shape();
    int seats = shape.variant().seats();
    row = new int[shape.inputs()];
    input = new float[shape.inputs()];
    state = new float[shape.layers()][shape.hidden()];
    inputGates = new float[shape.gates()];
    stateGates = new float[shape.gates()];
    projected = new float[shape.hidden()];
    logits = new double[shape.marginals()];
    marginals = new double[shape.marginals()];
    rowMeans = new double[seats];
    columnMeans = new double[seats];
    stride = shape.layers() * shape.hidden();
    stamps = new long[STATE_SLOTS];
    states = new float[STATE_SLOTS * stride];
    mask = STATE_SLOTS - 1;
  }

  public static GruWorkspace of(GruSinkhorn model) {
    return new GruWorkspace(model);
  }

  /** 状態のメモに当たり、履歴を読み直さずに済んだ回数。 */
  public long stateHits() {
    return stateHits;
  }

  /** 状態がメモに無く、履歴を最初から読んだ回数。 */
  public long stateMisses() {
    return stateMisses;
  }

  /** 指定したバージョンの履歴を読んだ後の状態がメモにあれば、それを今の状態に写して true を返す。 */
  boolean restore(long stamp) {
    int slot = slot(stamp);
    if (stamps[slot] != stamp) {
      stateMisses++;
      return false;
    }
    stateHits++;
    int base = slot * stride;
    for (float[] layer : state) {
      System.arraycopy(states, base, layer, 0, layer.length);
      base += layer.length;
    }
    return true;
  }

  /** 今の状態を、指定したバージョンの履歴を読んだ後の状態としてメモに入れる。 */
  void save(long stamp) {
    int slot = slot(stamp);
    stamps[slot] = stamp;
    int base = slot * stride;
    for (float[] layer : state) {
      System.arraycopy(layer, 0, states, base, layer.length);
      base += layer.length;
    }
  }

  private int slot(long stamp) {
    return (int) SeedMixer.mix64(stamp) & mask;
  }
}
