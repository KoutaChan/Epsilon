package com.epsilon.engine;

import java.util.AbstractList;
import java.util.Objects;
import java.util.RandomAccess;

/** 最大3席の同時判断を保持する、エンジン所有のバッファ。呼び出し元は判断の処理中だけ借用する。 */
final class EngineDecisionBatch extends AbstractList<EngineDecisionPoint> implements RandomAccess {

  private static final int MAX_DECISIONS = 3;

  private final EngineDecisionPoint[] points = new EngineDecisionPoint[MAX_DECISIONS];
  private int size;

  EngineDecisionBatch() {
    for (int index = 0; index < points.length; index++) {
      points[index] = new EngineDecisionPoint();
    }
  }

  EngineDecisionPoint append(long id, EngineDecisionKind kind, int player) {
    if (size == points.length) {
      throw new IllegalStateException("decision capacity exceeded: " + points.length);
    }
    EngineDecisionPoint point = points[size++];
    point.bind(id, kind, player);
    return point;
  }

  void discardLast() {
    if (size == 0) {
      throw new IllegalStateException("decision batch is empty");
    }
    points[--size].clear();
  }

  @Override
  public EngineDecisionPoint get(int index) {
    Objects.checkIndex(index, size);
    return points[index];
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public void clear() {
    for (int index = 0; index < size; index++) {
      points[index].clear();
    }
    size = 0;
  }

  void copyFrom(EngineDecisionBatch source) {
    clear();
    for (int index = 0; index < source.size; index++) {
      EngineDecisionPoint sourcePoint = source.points[index];
      EngineDecisionPoint target =
          append(sourcePoint.id(), sourcePoint.kind(), sourcePoint.player());
      target.mutableLegalActions().copyFrom(sourcePoint.mutableLegalActions());
      target.copyImmediateWinFrom(sourcePoint);
    }
  }
}
