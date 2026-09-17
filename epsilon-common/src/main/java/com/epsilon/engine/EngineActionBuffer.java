package com.epsilon.engine;

import com.epsilon.core.Action;
import java.util.AbstractList;
import java.util.List;
import java.util.Objects;
import java.util.RandomAccess;

/** エンジンが所有し、判断境界ごとに再利用する固定長の合法手バッファ。 */
public final class EngineActionBuffer extends AbstractList<Action> implements RandomAccess {

  private static final int MAX_ACTIONS = 32;

  private final short[] actionIds = new short[MAX_ACTIONS];
  private final List<Action> borrowedView = new BorrowedView();
  private int size;

  public EngineActionBuffer() {}

  @Override
  public Action get(int index) {
    Objects.checkIndex(index, size);
    return Action.fromIndex(actionIds[index] & 0xffff);
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public boolean contains(Object candidate) {
    if (!(candidate instanceof Action action)) {
      return false;
    }
    int actionId = action.toIndex();
    for (int index = 0; index < size; index++) {
      if ((actionIds[index] & 0xffff) == actionId) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean add(Action action) {
    if (size == actionIds.length) {
      throw new IllegalStateException("legal action capacity exceeded: " + actionIds.length);
    }
    actionIds[size++] = (short) Objects.requireNonNull(action, "action").toIndex();
    return true;
  }

  @Override
  public Action remove(int index) {
    Objects.checkIndex(index, size);
    Action removed = get(index);
    for (int source = index + 1; source < size; source++) {
      actionIds[source - 1] = actionIds[source];
    }
    size--;
    return removed;
  }

  @Override
  public void clear() {
    size = 0;
  }

  List<Action> borrowedView() {
    return borrowedView;
  }

  void copyFrom(EngineActionBuffer source) {
    clear();
    for (int index = 0; index < source.size; index++) {
      actionIds[size++] = source.actionIds[index];
    }
  }

  private final class BorrowedView extends AbstractList<Action> implements RandomAccess {

    @Override
    public Action get(int index) {
      return EngineActionBuffer.this.get(index);
    }

    @Override
    public int size() {
      return EngineActionBuffer.this.size;
    }

    @Override
    public boolean contains(Object candidate) {
      return EngineActionBuffer.this.contains(candidate);
    }
  }
}
