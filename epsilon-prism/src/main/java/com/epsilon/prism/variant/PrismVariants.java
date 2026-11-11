package com.epsilon.prism.variant;

import com.epsilon.mahjong.rules.Variant;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.ServiceLoader;

/** ServiceLoader で登録された、ルールごとの PRISM のモジュール。最初に使うときに一度だけ読み込む。 */
final class PrismVariants {

  static final Map<Variant, PrismVariant> LOADED = load();

  private PrismVariants() {}

  private static Map<Variant, PrismVariant> load() {
    Map<Variant, PrismVariant> modules = new EnumMap<>(Variant.class);
    for (PrismVariant module : ServiceLoader.load(PrismVariant.class, PrismVariant.class.getClassLoader())) {
      if (modules.putIfAbsent(module.variant(), module) != null) {
        throw new IllegalStateException("Two PRISM modules for " + module.variant() + " on the classpath");
      }
    }
    return Collections.unmodifiableMap(modules);
  }
}
