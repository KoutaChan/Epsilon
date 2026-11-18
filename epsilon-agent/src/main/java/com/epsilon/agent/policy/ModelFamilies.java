package com.epsilon.agent.policy;

import java.util.EnumMap;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;

/** 系列から、方策を開く入口（{@link ModelFamily}）を引く表。 */
public final class ModelFamilies {
  private static ModelFamilies installed;
  private final Map<Series, ModelFamily> bySeries;

  private ModelFamilies(Map<Series, ModelFamily> bySeries) {
    this.bySeries = bySeries;
  }

  /**
   * クラスパスの ServiceLoader から集める。
   *
   * @throws IllegalStateException 同じ系列の入口が 2 つあるとき
   */
  public static synchronized ModelFamilies load() {
    if (installed == null) installed = of(ServiceLoader.load(ModelFamily.class));
    return installed;
  }

  /**
   * 渡した入口から表を作る。
   *
   * @throws IllegalStateException 同じ系列の入口が 2 つあるとき
   */
  public static ModelFamilies of(Iterable<? extends ModelFamily> families) {
    Map<Series, ModelFamily> found = new EnumMap<>(Series.class);
    for (ModelFamily family : families) {
      ModelFamily previous = found.putIfAbsent(family.series(), family);
      if (previous != null)
        throw new IllegalStateException(
            "Two model families serve "
                + family.series()
                + ": "
                + previous.getClass().getName()
                + " and "
                + family.getClass().getName());
    }
    return new ModelFamilies(found);
  }

  public Optional<ModelFamily> find(Series series) {
    return Optional.ofNullable(bySeries.get(series));
  }
}
