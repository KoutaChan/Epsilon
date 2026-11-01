package com.epsilon.prism.model;

import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.RowColumn;

/** 1 塊の数と、行・手の索引のホストの写し（索引 {@link BatchIndex} と損失の入力 {@link LossInputs} をホストで作る元）。 */
interface HostColumns {

  int rows();

  int events();

  int transitions();

  int actions();

  int row(int row, RowColumn column);

  int action(int action, ActionColumn column);
}
