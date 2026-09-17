package com.epsilon.engine;

import com.epsilon.core.Action;
import java.util.AbstractList;
import java.util.Objects;
import java.util.RandomAccess;

/** 次に局面へ反映する行動の選択を格納する、呼び出し元所有の固定容量バッファ。 */
public final class EngineSelectionBuffer extends AbstractList<EngineDecisionSelection>
    implements RandomAccess {

  private static final int MAX_SELECTIONS = 3;

  private final EngineDecisionSelection[] selections = new EngineDecisionSelection[MAX_SELECTIONS];
  private int size;

  public EngineSelectionBuffer() {
    for (int index = 0; index < selections.length; index++) {
      selections[index] = new EngineDecisionSelection();
    }
  }

  public void add(long decisionId, Action action) {
    if (size == selections.length) {
      throw new IllegalStateException("selection capacity exceeded: " + selections.length);
    }
    selections[size++].bind(decisionId, action);
  }

  @Override
  public EngineDecisionSelection get(int index) {
    Objects.checkIndex(index, size);
    return selections[index];
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public void clear() {
    for (int index = 0; index < size; index++) {
      selections[index].clear();
    }
    size = 0;
  }
}
