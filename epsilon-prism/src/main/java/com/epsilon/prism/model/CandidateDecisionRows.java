package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import java.util.Arrays;

/**
 * 候補の補助ヘッドに決定行を候補の順に渡す。eager の学習では行を直接集め、同じ行への勾配は集める演算の逆伝播が足す。Attention の記録が
 * ある学習（専用カーネル）では、同じ行への勾配を候補の公開順に FP32 で足し、決定行の型に一度だけ丸める。どちらでも、同じ行の候補が
 * 無ければ行を直接集める。載せるインデックスは、使う集め方の分だけにする。
 */
final class CandidateDecisionRows {
  static final CandidateDecisionRows EMPTY =
      new CandidateDecisionRows(0, null, new Staging.Section[0], null);

  private final int count;
  private final Staging.Section rows;
  private final Staging.Section[] layers;
  private final Staging.Section published;

  private CandidateDecisionRows(
      int count, Staging.Section rows, Staging.Section[] layers, Staging.Section published) {
    this.count = count;
    this.rows = rows;
    this.layers = layers;
    this.published = published;
  }

  /**
   * 候補の決定行を載せる。公開順に足すときは、同じ行の最後の候補を第 0 層に置き、先に公開した候補ほど深い層に置く。
   *
   * @param staging 載せる先
   * @param rowCount バッチの行数
   * @param rows 候補ごとの決定行（候補の公開順）
   * @param ordered 同じ行への勾配を公開順に足す集め方にするか（Attention の記録がある学習なら true）
   * @return 載せた決定行
   */
  static CandidateDecisionRows stage(Staging staging, int rowCount, int[] rows, boolean ordered) {
    if (rows.length == 0) {
      return EMPTY;
    }
    int[] counts = new int[rowCount];
    int depth = 0;
    for (int row : rows) {
      depth = Math.max(depth, ++counts[row]);
    }
    if (!ordered || depth == 1) {
      var direct =
          staging.add(Arrays.stream(rows).asLongStream().toArray(), new Shape(rows.length));
      return new CandidateDecisionRows(rows.length, direct, new Staging.Section[0], null);
    }
    int[] sizes = new int[depth];
    for (int candidates : counts) {
      for (int layer = 0; layer < candidates; layer++) {
        sizes[layer]++;
      }
    }
    long[][] parents = new long[depth][];
    int[] starts = new int[depth];
    for (int layer = 0, start = 0; layer < depth; layer++) {
      starts[layer] = start;
      parents[layer] = new long[sizes[layer]];
      start += sizes[layer];
    }

    Arrays.fill(counts, 0);
    int[] next = new int[depth];
    int[] previous = new int[rowCount];
    long[] published = new long[rows.length];
    for (int candidate = rows.length - 1; candidate >= 0; candidate--) {
      int row = rows[candidate];
      int layer = counts[row]++;
      int position = next[layer]++;
      parents[layer][position] = layer == 0 ? row : previous[row];
      previous[row] = position;
      published[candidate] = starts[layer] + position;
    }
    Staging.Section[] layers = new Staging.Section[depth];
    for (int layer = 0; layer < depth; layer++) {
      layers[layer] = staging.add(parents[layer], new Shape(sizes[layer]));
    }
    return new CandidateDecisionRows(
        rows.length, null, layers, staging.add(published, new Shape(rows.length)));
  }

  /**
   * 決定行を候補の順に並べる。載せたときの集め方で集める。
   *
   * @param context 順伝播の文脈
   * @param decision 決定行 [行, H]
   * @return 候補ごとの決定行 [候補, H]（決定行と同じ型）
   */
  NDArray gather(Context context, NDArray decision) {
    if (count == 0) {
      return context.slice(decision, 0, 0);
    }
    if (layers.length == 0) {
      return NDArrays.gatherRows(decision, rows.array());
    }
    return context.scoped(
        scope -> {
          NDArray active = scope.toFloat(decision);
          NDList outputs = new NDList();
          for (Staging.Section layer : layers) {
            // 各層の添字に重複はない。逆伝播では、先に公開した候補の部分和にこの候補の勾配が加わる。
            active = NDArrays.gatherRows(active, layer.array());
            outputs.add(active);
          }
          NDArray ordered = NDArrays.gatherRows(NDArrays.concat(outputs), published.array());
          return scope.cast(ordered, decision.getDataType());
        },
        decision);
  }
}
