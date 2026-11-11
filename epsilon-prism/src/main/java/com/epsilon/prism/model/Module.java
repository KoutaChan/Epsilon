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

/**
 * ネットワークのモジュールの基底クラス。パラメーターを、形と初期値と一緒に登録する。モジュールは型つきの入口のメソッドだけを持ち、DJL の
 * forward は使わない。
 *
 * <p>初期値は、Linear の重みが DJL の既定（He の正規分布）、埋め込みの表が N(0, 0.02²)、ゲインが 1、バイアスと最後の層が 0。
 */
public abstract class Module extends AbstractBlock {

  /** 埋め込みの表の初期値の標準偏差。 */
  private static final float TABLE_SIGMA = 0.02f;

  /** Linear の重みの初期値（DJL の Linear の既定と同じ）。 */
  private static final Initializer WEIGHT =
      new XavierInitializer(
          XavierInitializer.RandomType.GAUSSIAN, XavierInitializer.FactorType.IN, 2);

  /**
   * Linear の重み（[出力, 入力]）を登録する。
   *
   * @param name 名前
   * @param shape 形
   * @return パラメーター
   */
  protected final Parameter weight(String name, Shape shape) {
    return register(name, Parameter.Type.WEIGHT, shape, WEIGHT);
  }

  /**
   * 0 で初期化するパラメーター（バイアス、最後の層、関係のバイアスの表、コードごとのキーの表など）を登録する。
   *
   * @param name 名前
   * @param shape 形
   * @return パラメーター
   */
  protected final Parameter zeros(String name, Shape shape) {
    return register(name, Parameter.Type.OTHER, shape, new ConstantInitializer(0));
  }

  /**
   * 埋め込みの表（[語彙, 次元]、初期値は N(0, 0.02²)）を登録する。
   *
   * @param name 名前
   * @param rows 語彙の数
   * @param dimension 次元
   * @return パラメーター
   */
  protected final Parameter table(String name, int rows, int dimension) {
    return register(name, Parameter.Type.WEIGHT, new Shape(rows, dimension), new NormalInitializer(TABLE_SIGMA));
  }

  /**
   * RMSNorm のゲイン（幅 H、1 で初期化する）を登録する。
   *
   * @param name 名前
   * @param hidden 幅 H
   * @return パラメーター
   */
  protected final Parameter gain(String name, int hidden) {
    return register(name, Parameter.Type.GAMMA, new Shape(hidden), Initializer.ONES);
  }

  /**
   * 0 で初期化する 1 次元のバイアスを登録する。
   *
   * @param name 名前
   * @param size 幅
   * @return パラメーター
   */
  protected final Parameter bias(String name, int size) {
    return register(name, Parameter.Type.BIAS, new Shape(size), Initializer.ZEROS);
  }

  private Parameter register(String name, Parameter.Type type, Shape shape, Initializer initializer) {
    return addParameter(
        Parameter.builder()
            .setName(name)
            .setType(type)
            .optShape(shape)
            .optInitializer(initializer)
            .build());
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
