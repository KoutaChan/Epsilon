package com.epsilon.classic;

import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * アーキテクチャによって有るかどうかが変わる定数。アーキテクチャごとの並びは、そのアーキテクチャが持つ定数だけを宣言の順に並べたもので、
 * オフセットはその並びの中の位置から求める。
 */
public interface ArchitectureScoped {

  /** この定数を持つアーキテクチャ。 */
  Set<Architecture> architectures();

  default boolean in(Architecture architecture) {
    return architectures().contains(architecture);
  }

  /**
   * 定数を持つアーキテクチャの集合を作る。
   *
   * @param only 定数を持つアーキテクチャ。空なら全てのアーキテクチャが持つ
   * @return 変更できない集合
   */
  static Set<Architecture> scope(Architecture... only) {
    EnumSet<Architecture> set =
        only.length == 0 ? EnumSet.allOf(Architecture.class) : EnumSet.copyOf(List.of(only));
    return Collections.unmodifiableSet(set);
  }
}
