package com.epsilon.prism.model;

import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputFormat;

/**
 * 1 つのバッチの件数と、行・手の列をホストに写したもの。インデックス（{@link BatchIndex}）と損失の入力（{@link LossInputs}）を
 * ホストで作るときに読む。
 */
interface HostColumns {

  /** バッチの入力の形式。 */
  InputFormat format();

  int rows();

  int events();

  int transitions();

  int actions();

  int row(int row, RowColumn column);

  int action(int action, ActionColumn column);
}
