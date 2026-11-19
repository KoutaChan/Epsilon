package com.epsilon.prism.grp;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.grp.PhiCacheSizes;

/**
 * Φ のキャッシュの大きさ（{@link PhiCacheSizes}）。どちらもスロットの数で、2 の冪。大きさは速さとメモリにだけ効き、Φ の値には効かない。
 *
 * @param thread スレッドごとのキャッシュのスロットの数
 * @param shared 全てのスレッドで共有するキャッシュのスロットの数
 */
public record PhiCacheSettings(
    @Key(doc = "スレッドごとの Φ のキャッシュのスロットの数（2 の冪）", value = "4096", change = Change.MACHINE)
        @Range(min = 1, max = LateSettings.MAX_CACHE)
        int thread,
    @Key(doc = "全てのスレッドで共有する Φ のキャッシュのスロットの数（2 の冪）", value = "65536", change = Change.MACHINE)
        @Range(min = 1, max = LateSettings.MAX_CACHE)
        int shared) {

  public PhiCacheSettings {
    if (Integer.bitCount(thread) != 1) {
      throw new InvalidSetting(
          "thread", "the thread Phi cache needs a power of two slots, got " + thread);
    }
    if (Integer.bitCount(shared) != 1) {
      throw new InvalidSetting(
          "shared", "the shared Phi cache needs a power of two slots, got " + shared);
    }
  }

  /** キャッシュの大きさを返す。 */
  public PhiCacheSizes sizes() {
    return new PhiCacheSizes(thread, shared);
  }
}
