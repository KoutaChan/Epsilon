package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.CandidateCodes;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;

/** 遷移の牌コードから学ぶゲートで牌の射影を重み付けし、牌種方向の総和を候補の残差に足す。 */
final class CandidateCount extends Module {
  private final int width;
  private final int firstTile;
  private final int firstCode;
  private final Parameter gate;
  private final Parameter tile;
  private final Parameter output;

  CandidateCount(PrismShape shape, InputFormat format) {
    width = CandidateCodes.COUNT_WIDTH;
    firstTile = TokenType.TILE.first(format);
    firstCode = format.tileKeyCode().first();
    gate = table("gate", format.keys().codes() - firstCode, width);
    tile = weight("tile", new Shape(width, shape.hidden()));
    Shape projection = new Shape(shape.hidden(), width);
    output = weight("output", projection);
  }

  public NDArray add(
      Context context, NetworkInputs inputs, BatchIndex index, NDArray states, NDArray residual) {
    NDArray tiles = states.get(new NDIndex(":, {}:", firstTile));
    NDArray projected = context.linear(tiles, tile, null);
    NDArray ids =
        inputs
            .get(Tensor.TRANSITION_CODE)
            .get(new NDIndex(":, {}:{}", firstTile, inputs.format().fixedTokens()))
            .toType(DataType.INT32, false)
            .sub(firstCode);
    NDArray rows = index.transitionRows.array();
    NDArray count = context.candidateCount(projected, gate, rows, ids);
    if (count == null) {
      NDArray shared = NDArrays.gatherRows(projected, rows);
      NDArray gates = context.rows(gate, ids.reshape(-1)).reshape(ids.getShape().add(width));
      count = context.toFloat(shared).mul(Activation.sigmoid(gates)).sum(new int[] {1});
    }
    return residual.add(context.linear(context.toCompute(count), output, null));
  }
}
