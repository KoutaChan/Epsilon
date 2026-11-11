package com.epsilon.prism.input;

import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 牌のトークンの並び。そのルールで使う牌の種類を、種類の番号の順に並べたもの。四人打ちは 34 個（トークンの番号は種類の番号と同じ）、
 * 三人打ちは 2〜8 萬を除いた 27 個。入力のうち、牌の番号を書くフィールド、関係の属性の牌、キーのコードで牌を書く範囲は、どれも正準の
 * 色の順に直した牌をこのトークンの番号で書く。
 *
 * <p>入れ替えてよい色（{@link #movableSuits}）は、1〜9 が全てそのルールで使われる数牌の色。どの色も種類の並びが同じなので、入れ替えても
 * トークンの番号の並びは変わらない。
 */
public final class TileTokens {

  private static final TileTokens[] BY_VARIANT = load();

  private final Variant variant;
  private final int[] kinds;
  private final int[] tokens;
  private final int[] relations;
  private final int movableSuits;

  private TileTokens(Variant variant) {
    this.variant = variant;
    long set = variant.tiles().kinds();
    kinds = new int[Long.bitCount(set)];
    tokens = new int[Tiles.KINDS];
    Arrays.fill(tokens, -1);
    int token = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (Tiles.contains(set, kind)) {
        kinds[token] = kind;
        tokens[kind] = token++;
      }
    }
    relations = new int[kinds.length * kinds.length];
    for (int query = 0; query < kinds.length; query++) {
      for (int key = 0; key < kinds.length; key++) {
        relations[query * kinds.length + key] = MahjongTileRelation.between(kinds[query], kinds[key]).ordinal();
      }
    }
    int movable = 0;
    for (Tiles.Group group : Tiles.Group.values()) {
      if (group.suited() && (set & Tiles.range(group.first(), group.last())) == Tiles.range(group.first(), group.last())) {
        movable |= 1 << group.ordinal();
      }
    }
    movableSuits = movable;
  }

  /**
   * そのルールの牌のトークンの並びを返す。
   *
   * @param variant ルール（四人打ちか三人打ち）
   * @return 並び
   */
  public static TileTokens of(Variant variant) {
    return BY_VARIANT[variant.ordinal()];
  }

  public Variant variant() {
    return variant;
  }

  /** トークンの数。 */
  public int count() {
    return kinds.length;
  }

  /**
   * トークンの牌の種類を返す。
   *
   * @param token トークンの番号
   * @return 種類
   */
  public int kind(int token) {
    return kinds[Objects.checkIndex(token, kinds.length)];
  }

  /**
   * 牌の種類をトークンの番号にする。
   *
   * @param kind 牌の種類（負の値はそのまま返す）
   * @return トークンの番号
   * @throws IllegalArgumentException そのルールで使わない種類のとき
   */
  public int token(int kind) {
    if (kind < 0) {
      return kind;
    }
    int token = tokens[kind];
    if (token < 0) {
      throw new IllegalArgumentException(variant + " has no tile kind " + kind);
    }
    return token;
  }

  /**
   * 牌の種類のマスクをトークンのマスクにする。
   *
   * @param kinds 牌の種類のマスク（そのルールで使わない種類を含まないこと）
   * @return ビット t がトークン t のマスク
   */
  public long mask(long kinds) {
    long mask = 0L;
    for (long rest = kinds; rest != 0L; rest &= rest - 1) {
      mask |= 1L << token(Long.numberOfTrailingZeros(rest));
    }
    return mask;
  }

  /**
   * トークンの組ごとに、牌の関係の種類（{@link MahjongTileRelation} の ordinal）を返す。添字は「クエリ × トークンの数 ＋ キー」。
   *
   * @return 表の写し
   */
  public int[] relations() {
    return relations.clone();
  }

  /** 入れ替えてよい数牌の色のビット。ビット g は、{@link Tiles.Group} の ordinal が g の色。 */
  public int movableSuits() {
    return movableSuits;
  }

  private static TileTokens[] load() {
    Variant[] variants = Variant.values();
    TileTokens[] tables = new TileTokens[variants.length];
    for (Variant variant : variants) {
      tables[variant.ordinal()] = new TileTokens(variant);
    }
    return tables;
  }
}
