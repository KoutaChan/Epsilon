package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;
import java.util.List;

/**
 * 関数を保って網を広げる。白紙で作った広い網に、狭い網の値を名前で写し、新しい添字は {@link WidthPlan} の軸の役で決める。
 *
 * <ul>
 *   <li>古い添字だけの要素は狭い網の値。
 *   <li>新しい添字を持つ要素は、どれかの新しい軸が書く側（{@link Role#writes()}）なら 0、読む側だけなら広い網の白紙の値。
 *   <li>RMSNorm の利得は古い成分を √(H/H′) 倍し、新しい成分を √(H/H′) にする（ε は形が H/H′ 倍にする）。
 * </ul>
 *
 * <p>流れに書く行が新しい次元で 0 なので流れの新しい次元は 0 のまま、新しい頭と素子は出の重みが 0 なので出力に効かない。実数では関数が
 * 厳密に保たれ、読む側が乱数なので 1〜2 歩で新しいパラメーターに勾配が届く。層の数と名前の例外に依らない。
 */
public final class PrismWiden {

  private PrismWiden() {}

  /**
   * 広い形の網を白紙で作り、narrow の関数を写して返す。
   *
   * @param narrow 狭い網
   * @param shape 広い形（深さと前線の腕は同じ、幅はどれも以上）
   * @param manager 広い網のパラメーターの管理元
   * @return 広い網
   * @throws IllegalArgumentException 深さか前線の腕が違うか、幅が狭くなるとき
   */
  public static PrismNetwork widen(PrismNetwork narrow, PrismShape shape, NDManager manager) {
    PrismShape from = narrow.shape();
    if (shape.depth() != from.depth()
        || shape.frontier() != from.frontier()
        || shape.hidden() < from.hidden()
        || shape.ffn() < from.ffn()
        || shape.gate() < from.gate()) {
      throw new IllegalArgumentException("cannot widen " + from + " to " + shape);
    }
    PrismNetwork wide = new PrismNetwork(shape);
    wide.initialize(manager);
    copy(narrow, wide);
    return wide;
  }

  /**
   * narrow の関数を、白紙で作った wide に写す。
   *
   * @param narrow 狭い網
   * @param wide 白紙の広い網
   */
  public static void copy(PrismNetwork narrow, PrismNetwork wide) {
    float gainScale = (float) Math.sqrt((double) narrow.shape().hidden() / wide.shape().hidden());
    WidthPlan plan = wide.widthPlan();
    PairList<String, Parameter> sources = narrow.getParameters();
    PairList<String, Parameter> targets = wide.getParameters();
    if (sources.size() != targets.size()) {
      throw new IllegalArgumentException("networks have different parameter lists");
    }
    for (Pair<String, Parameter> pair : targets) {
      Parameter source = sources.get(pair.getKey());
      if (source == null) {
        throw new IllegalArgumentException("missing parameter " + pair.getKey());
      }
      NDArray from = source.getArray();
      NDArray to = pair.getValue().getArray();
      float[] values =
          map(
              pair.getKey(),
              from.toFloatArray(),
              from.getShape(),
              to.toFloatArray(),
              to.getShape(),
              plan.axes(pair.getKey()),
              gainScale);
      to.set(values);
    }
  }

  /** 1 つのパラメーターの値を写す。 */
  private static float[] map(
      String name,
      float[] old,
      Shape oldShape,
      float[] fresh,
      Shape newShape,
      List<Axis> axes,
      float gainScale) {
    int dimensions = newShape.dimension();
    int[][] sources = new int[dimensions][];
    for (int axis = 0; axis < dimensions; axis++) {
      sources[axis] =
          sourceIndices(name, axes.get(axis), (int) oldShape.get(axis), (int) newShape.get(axis));
    }
    boolean gain = dimensions == 1 && axes.get(0).role() == Role.GAIN;
    float[] result = new float[fresh.length];
    int[] position = new int[dimensions];
    for (int element = 0; element < result.length; element++) {
      int oldIndex = 0;
      boolean known = true;
      boolean zero = false;
      for (int axis = 0; axis < dimensions; axis++) {
        int source = sources[axis][position[axis]];
        if (source < 0) {
          known = false;
          zero |= axes.get(axis).role().writes();
        } else {
          oldIndex = oldIndex * (int) oldShape.get(axis) + source;
        }
      }
      if (gain) {
        result[element] = known ? old[oldIndex] * gainScale : gainScale;
      } else if (known) {
        result[element] = old[oldIndex];
      } else {
        result[element] = zero ? 0f : fresh[element];
      }
      for (int axis = dimensions - 1; axis >= 0; axis--) {
        if (++position[axis] < newShape.get(axis)) {
          break;
        }
        position[axis] = 0;
      }
    }
    return result;
  }

  /** 新しい添字ごとの古い添字（新しい添字は −1）。塊ごとに古い添字を塊の先頭に写す。 */
  private static int[] sourceIndices(String name, Axis axis, int oldSize, int newSize) {
    if (newSize < oldSize || (axis.role().fixed() && newSize != oldSize)) {
      throw new IllegalArgumentException(name + " cannot change a " + axis.role() + " axis");
    }
    int blocks = axis.blocks();
    int oldBlock = oldSize / blocks;
    int newBlock = newSize / blocks;
    int[] sources = new int[newSize];
    for (int index = 0; index < newSize; index++) {
      int block = index / newBlock;
      int offset = index % newBlock;
      sources[index] = offset < oldBlock ? block * oldBlock + offset : -1;
    }
    return sources;
  }
}
