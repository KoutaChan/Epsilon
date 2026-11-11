package com.epsilon.grp.train;

import com.epsilon.grp.GrpKey;
import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * GRP の学習に使う行を、値の種類ごとに 1 本の配列で持つ。1 行は半荘が続く局末 1 つで、卓のキー（GrpKey）、実際の順列（相対席）、
 * コホートの番号（範囲の最初から何番目か）を持つ。1 行は 10 バイトで、行ごとにオブジェクトは作らない。行の重みはコホートから引く
 * （RecencyWindow）。学習用の行と、早期終了に使う検証用の行は、別々の GrpRows に持つ。行の目標は実際の順列の one-hot。
 */
public final class GrpRows implements TrainingLoop.Rows {

  /** 検証用の行を取るコホートの数。範囲の新しい方から数える。 */
  public static final int STOP_COHORTS = 2;

  private final Variant variant;
  private final CohortWeights weights;
  final long[] keys;
  final byte[] permutations;
  final byte[] cohorts;
  private final int size;
  private final long excluded;

  private GrpRows(Variant variant, CohortWeights weights, Column column) {
    this.variant = variant;
    this.weights = weights;
    keys = Arrays.copyOf(column.keys, column.size);
    permutations = Arrays.copyOf(column.permutations, column.size);
    cohorts = Arrays.copyOf(column.cohorts, column.size);
    size = column.size;
    excluded = column.excluded;
  }

  /** 範囲にあるコホートの半荘から、学習用の行と検証用の行を作る。 */
  public static Builder builder(Variant variant, RecencyWindow window, GateCohort gate) {
    return new Builder(variant, window, gate);
  }

  public Variant variant() {
    return variant;
  }

  public RecencyWindow window() {
    return weights.window();
  }

  @Override
  public int size() {
    return size;
  }

  public long key(int row) {
    return keys[Objects.checkIndex(row, size)];
  }

  /** 実際の順列の番号。順列は相対席で表し、番号の付け方は RankPermutations に従う。 */
  public int permutation(int row) {
    return permutations[Objects.checkIndex(row, size)];
  }

  public int cohort(int row) {
    return weights.cohort(cohorts[Objects.checkIndex(row, size)]);
  }

  public float weight(int row) {
    return weights.weight(cohorts[Objects.checkIndex(row, size)]);
  }

  /** 卓を GrpKey で表せないために外した局末の数。 */
  public long excluded() {
    return excluded;
  }

  @Override
  public double totalWeight() {
    return weights.total(cohorts, size);
  }

  /** 行の重み。学習のループの中で行ごとに呼ぶので、添字は確かめない。 */
  float weightOf(int row) {
    return weights.weight(cohorts[row]);
  }

  @Override
  public String toString() {
    return "GrpRows[" + variant + " " + window() + " rows=" + size + " excluded=" + excluded + "]";
  }

  /** 範囲の学習用の行と検証用の行。 */
  public record Split(GrpRows train, GrpRows stop) {}

  /** 範囲のコホートを 1 つずつ足して行を作る。半荘の分け方は CohortSplit に従う。 */
  public static final class Builder {

    private final Variant variant;
    private final CohortSplit split;
    private final Column train = new Column();
    private final Column stop = new Column();

    private Builder(Variant variant, RecencyWindow window, GateCohort gate) {
      this.variant = variant;
      split = new CohortSplit(variant, window, gate);
    }

    /** 範囲の外のコホート、既に足したコホート、ルールの違う半荘を渡すと IllegalArgumentException を投げる。 */
    public Builder add(int cohort, List<GameRecord> games) {
      CohortSplit.Target[] targets = split.accept(cohort, games);
      byte offset = split.weights().offset(cohort);
      for (int index = 0; index < targets.length; index++) {
        switch (targets[index]) {
          case TRAIN -> train.add(games.get(index), offset);
          case STOP -> stop.add(games.get(index), offset);
          case NONE -> {
            // 行にしない。
          }
        }
      }
      return this;
    }

    public Split build() {
      return new Split(
          new GrpRows(variant, split.weights(), train), new GrpRows(variant, split.weights(), stop));
    }
  }

  /** 作っている途中の、学習用または検証用の行。配列が埋まれば長さを倍にし、{@code build} のときに行の数まで切り詰める。 */
  private static final class Column {

    private long[] keys = new long[1 << 12];
    private byte[] permutations = new byte[keys.length];
    private byte[] cohorts = new byte[keys.length];
    private int size;
    private long excluded;

    void add(GameRecord game, byte cohort) {
      List<GrpState> handEnds = game.handEnds();
      for (int handEnd = 0; handEnd < handEnds.size(); handEnd++) {
        long key = handEnds.get(handEnd).key();
        if (key == GrpKey.NONE) {
          excluded++;
          continue;
        }
        if (size == keys.length) {
          int capacity = 2 * keys.length;
          keys = Arrays.copyOf(keys, capacity);
          permutations = Arrays.copyOf(permutations, capacity);
          cohorts = Arrays.copyOf(cohorts, capacity);
        }
        keys[size] = key;
        permutations[size] = (byte) game.realizedPermutation(handEnd);
        cohorts[size] = cohort;
        size++;
      }
    }
  }
}
