package com.epsilon.prism.model;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 網の部品の土台。パラメーターを形・初期値・軸の役（{@link WidthPlan}）とともに登録する。部品は型つきの入口だけを持ち、DJL の
 * forward は使わない。
 *
 * <p>初期値: Linear の重みは DJL の既定（He の正規分布）、埋め込みの表は N(0, 0.02²)、利得は 1、偏りと最後の層は 0。
 */
public abstract class Module extends AbstractBlock {

  /** 埋め込みの表の初期値の標準偏差。 */
  private static final float TABLE_SIGMA = 0.02f;

  /** Linear の重みの初期値（DJL の Linear の既定と同じ）。 */
  private static final Initializer WEIGHT =
      new XavierInitializer(
          XavierInitializer.RandomType.GAUSSIAN, XavierInitializer.FactorType.IN, 2);

  private final Map<Parameter, List<Axis>> axes = new LinkedHashMap<>();

  /**
   * Linear の重み（[出力, 入力]）を登録する。
   *
   * @param name 名前
   * @param shape 形
   * @param roles 軸ごとの役
   * @return パラメーター
   */
  protected final Parameter weight(String name, Shape shape, Axis... roles) {
    return register(name, Parameter.Type.WEIGHT, shape, WEIGHT, roles);
  }

  /**
   * 0 で始めるパラメーター（偏り、最後の層、関係の偏りの表、鍵の符号の表）を登録する。
   *
   * @param name 名前
   * @param shape 形
   * @param roles 軸ごとの役
   * @return パラメーター
   */
  protected final Parameter zeros(String name, Shape shape, Axis... roles) {
    return register(name, Parameter.Type.OTHER, shape, new ConstantInitializer(0), roles);
  }

  /**
   * 埋め込みの表（[語彙, 次元]、N(0, 0.02²)）を登録する。
   *
   * @param name 名前
   * @param rows 語彙
   * @param dimension 次元
   * @return パラメーター
   */
  protected final Parameter table(String name, int rows, int dimension) {
    return register(
        name,
        Parameter.Type.WEIGHT,
        new Shape(rows, dimension),
        new NormalInitializer(TABLE_SIGMA),
        Axis.of(Role.VOCAB),
        Axis.of(Role.FIXED));
  }

  /**
   * RMSNorm の利得（幅 H、1 で始める）を登録する。
   *
   * @param name 名前
   * @param hidden 幅 H
   * @return パラメーター
   */
  protected final Parameter gain(String name, int hidden) {
    return register(
        name, Parameter.Type.GAMMA, new Shape(hidden), Initializer.ONES, Axis.of(Role.GAIN));
  }

  /**
   * 0 で始める偏り（1 次元）を登録する。
   *
   * @param name 名前
   * @param size 幅
   * @param role 軸の役
   * @return パラメーター
   */
  protected final Parameter bias(String name, int size, Axis role) {
    return register(name, Parameter.Type.BIAS, new Shape(size), Initializer.ZEROS, role);
  }

  /**
   * 登録したパラメーターの軸の役を返す。
   *
   * @param parameter この部品のパラメーター
   * @return 軸の並び
   */
  final List<Axis> axes(Parameter parameter) {
    List<Axis> result = axes.get(parameter);
    if (result == null) {
      throw new IllegalArgumentException("parameter is not registered here: " + parameter);
    }
    return result;
  }

  private Parameter register(
      String name, Parameter.Type type, Shape shape, Initializer initializer, Axis... roles) {
    if (roles.length != shape.dimension()) {
      throw new IllegalArgumentException(name + " declares " + roles.length + " axes for " + shape);
    }
    for (int axis = 0; axis < roles.length; axis++) {
      if (shape.get(axis) % roles[axis].blocks() != 0) {
        throw new IllegalArgumentException(name + " axis " + axis + " does not split in blocks");
      }
    }
    Parameter parameter =
        addParameter(
            Parameter.builder()
                .setName(name)
                .setType(type)
                .optShape(shape)
                .optInitializer(initializer)
                .build());
    axes.put(parameter, List.of(roles));
    return parameter;
  }

  @Override
  protected final NDList forwardInternal(
      ParameterStore store, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use the typed entry point");
  }

  @Override
  public final Shape[] getOutputShapes(Shape[] inputShapes) {
    throw new UnsupportedOperationException("Shapes follow PrismShape");
  }
}
