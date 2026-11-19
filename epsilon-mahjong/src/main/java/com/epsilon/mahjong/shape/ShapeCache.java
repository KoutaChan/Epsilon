package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape.Block;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.EnumMap;
import java.util.Map;

/**
 * {@link ShapeAnalyzer#analyze} の結果を手の形ごとに覚え、スレッドの間で共有する表。結果は手の形（色ごとの枚数の符号と面子の数）だけで
 * 決まるので、覚えた値は {@link ShapeAnalyzer#analyze} と同じになる。表は直接写像で、同じ位置に入る別の形は上書きする。作った後は
 * メモリを確保しない。
 *
 * <p>位置ごとに版の番号・キー 2 つ・値を {@code long} で持つ。書き手は版を奇数にしてから書き、偶数に戻す（ほかの書き手が書いている
 * 位置には書かない）。読み手は版を読み、キーと値を読んだ後に版を読み直し、2 回の版が同じ偶数でキーが合ったときだけ値を使う。
 */
public final class ShapeCache {

  private static final Map<Variant, ShapeCache> SHARED = new EnumMap<>(Variant.class);

  /** 位置の数の 2 の対数。 */
  private static final int BITS = 16;

  private static final int STRIDE = 4;
  private static final VarHandle WORDS = MethodHandles.arrayElementVarHandle(long[].class);

  /** 萬子・筒子・索子の符号を 1 つに並べるときの重み（1 色の符号は 5 の 9 乗未満）。 */
  private static final long SUIT_RADIX = (long) Block.MAN.codes();

  private final ShapeAnalyzer analyzer;
  private final long[] words = new long[STRIDE << BITS];

  private ShapeCache(Variant variant) {
    analyzer = new ShapeAnalyzer(variant);
  }

  /**
   * ルールごとに 1 つの共有の表を返す。
   *
   * @param variant ルール（使う牌の種類で、有効牌が変わる）
   * @return そのルールの表
   */
  public static ShapeCache shared(Variant variant) {
    synchronized (SHARED) {
      return SHARED.computeIfAbsent(variant, ShapeCache::new);
    }
  }

  /**
   * {@link ShapeAnalyzer#analyze} と同じ結果を書く。表に同じ形があればその値を使い、なければ求めて表に入れる。
   *
   * @param hand 手（14 枚なら IllegalArgumentException を投げる）
   * @param out 結果を書く先
   */
  public void analyze(HandShape hand, ShapeAnalyzer.Analysis out) {
    long suits =
        hand.code(Block.MAN)
            + SUIT_RADIX * (hand.code(Block.PIN) + SUIT_RADIX * hand.code(Block.SOU));
    long rest = ((long) hand.code(Block.HONOR) << 3 | hand.melds()) + 1;
    long mixed = suits * 0x9E3779B97F4A7C15L ^ rest * 0xC2B2AE3D27D4EB4FL;
    int at = (int) ((mixed ^ mixed >>> 29) >>> (Long.SIZE - BITS)) * STRIDE;
    boolean concealed = hand.melds() == 0;
    long version = (long) WORDS.getAcquire(words, at);
    if ((version & 1) == 0 && words[at + 1] == suits && words[at + 2] == rest) {
      long packed = words[at + 3];
      VarHandle.acquireFence();
      if ((long) WORDS.getVolatile(words, at) == version) {
        out.unpack(packed, concealed);
        return;
      }
    }
    analyzer.analyze(hand, out);
    long current = (long) WORDS.getVolatile(words, at);
    if ((current & 1) == 0 && WORDS.compareAndSet(words, at, current, current + 1)) {
      words[at + 1] = suits;
      words[at + 2] = rest;
      words[at + 3] = out.pack();
      WORDS.setRelease(words, at, current + 2);
    }
  }
}
