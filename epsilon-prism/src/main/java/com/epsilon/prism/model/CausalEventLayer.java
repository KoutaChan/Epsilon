package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.input.InputFormat;

/** 層 0 で固定トークンは公開列全体、イベントは固定トークンを除く自分までの接頭辞だけを読む。 */
final class CausalEventLayer extends EncoderLayer {
  private final int fixed;

  CausalEventLayer(PrismShape.V4 shape, int layer, InputFormat format) {
    super(shape, layer, format);
    fixed = format.fixedTokens();
  }

  @Override
  int causalFixedTokens() {
    return fixed;
  }

  @Override
  NDArray eager(Context context, NDArray query, NDArray keyValue, NDArray mask, NDArray codes) {
    int tokens = Math.toIntExact(query.getShape().get(1));
    if (tokens == fixed) return super.eager(context, query, keyValue, mask, codes);
    NDArray table = packedTable(context);
    NDIndex fixedQueries = new NDIndex(":, :{}", fixed);
    NDList attended =
        new NDList(packed(query.get(fixedQueries), keyValue, mask, codes.get(fixedQueries), table));
    // 実イベントの手前にはパディングがない。短い行の末尾の出力は次の層のキーマスクで外れる。
    NDArray present =
        context
            .working()
            .ones(new Shape(mask.getShape().get(0), tokens - fixed), mask.getDataType());
    for (int at = fixed; at < tokens; at++) {
      int end = at + 1;
      NDArray q = query.get(new NDIndex(":, {}:{}", at, end));
      NDArray kv = keyValue.get(new NDIndex(":, {}:{}", fixed, end));
      NDArray pairs = codes.get(new NDIndex(":, {}:{}, {}:{}", at, end, fixed, end));
      attended.add(packed(q, kv, present.get(new NDIndex(":, :{}", end - fixed)), pairs, table));
    }
    return NDArrays.concat(attended, 1);
  }
}
