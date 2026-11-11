package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 形の表のファイル（リソースの {@link #RESOURCE}）に入るセクション。ファイルは、セクションを宣言の順に並べて gzip で圧縮したもの。
 * 各セクションは要素の数（int）と要素からなり、ビッグエンディアンで書く。表を作るプログラムはテストの側にある。
 *
 * <p>キーは、コードを同値類に分けたときの番号。同じキーのコードは、全ての (m, p)（m ≤ 4、p ≤ 1）について、面子 m 個と
 * 雀頭 p 個を作るのに足りない最小の枚数が同じ。形を作るときは、ブロックの各種類を 4 枚まで使う。違うブロックが同じ種類を使う
 * ことは無いので、手全体の足りない枚数は、ブロックごとの足りない枚数を (min, +) で合わせて求められる。通常の形では、色のキーを
 * {@link #TWO_SUITS}・{@link #THREE_SUITS} で合わせ、字牌のキーと一緒に {@link #DISTANCE} を引く。
 */
enum ShapeSection {
  /** 数牌 1 色のコード → キー。15 枚以上のコードは使わないので 0 を入れる。 */
  SUIT_KEY(Element.BYTE),
  /** 字牌のコード → 字牌のキー。15 枚以上のコードは使わないので 0 を入れる。 */
  HONOR_KEY(Element.BYTE),
  /**
   * 1m の枚数 ＋ 5 × 9m の枚数 → 数牌のキー。萬子が 1m・9m だけのルール（三人打ち）では、萬子のキーをこの表から引く。2〜8 萬が
   * 無いので 1m と 9m は順子にならない。例えば 1m 4 枚から面子 2 つを作るには 3 枚足りない（{@link #SUIT_KEY} なら 2m・3m を
   * 引けるので 2 枚）。
   */
  TERMINAL_MAN_KEY(Element.BYTE),
  /** [キー × 数牌のキーの数 ＋ キー] → 2 色のキー。 */
  TWO_SUITS(Element.BYTE),
  /** [2 色のキー × 数牌のキーの数 ＋ キー] → 3 色のキー。 */
  THREE_SUITS(Element.BYTE),
  /** [(3 色のキー × 字牌のキーの数 ＋ 字牌のキー) × 5 ＋ m] → 面子 m 個と雀頭 1 つを作るのに足りない最小の枚数。 */
  DISTANCE(Element.BYTE),
  /** 数牌 1 色のコード → 分解の状態。0 は分解が無いこと（面子と、高々 1 つの雀頭に余りなく分けられない）を表す。 */
  SUIT_STATE(Element.CHAR),
  /** 字牌のコード → 分解の状態。 */
  HONOR_STATE(Element.CHAR),
  /** [状態] → 最初の分解の番号。状態 s の分解は、FIRST[s] から FIRST[s + 1] − 1 まで。 */
  FIRST(Element.INT),
  /** [分解] → 雀頭の、ブロックの中での番号。雀頭が無ければ −1。 */
  PAIR(Element.BYTE),
  /**
   * [分解 × 4 ＋ i] → i 番目の面子。刻子はブロックの中での番号、順子は {@link #RUN} ＋ 先頭の番号、空きは −1。面子は番号の
   * 昇順に並び、同じ番号では刻子が先。
   */
  MELD(Element.BYTE);

  static final String RESOURCE = "/shape/hand-shape.bin";

  /** {@link #MELD} で、順子の先頭の番号に足す数。 */
  static final int RUN = Tiles.SUIT_SIZE;

  /** {@link #DISTANCE} の 1 行の幅。面子の数 0〜4 の 5 通り。 */
  static final int MELD_TARGETS = Variant.MAX_MELDS + 1;

  enum Element {
    BYTE(Byte.BYTES),
    CHAR(Character.BYTES),
    INT(Integer.BYTES);

    private final int bytes;

    Element(int bytes) {
      this.bytes = bytes;
    }

    int bytes() {
      return bytes;
    }
  }

  private final Element element;

  ShapeSection(Element element) {
    this.element = element;
  }

  Element element() {
    return element;
  }
}
