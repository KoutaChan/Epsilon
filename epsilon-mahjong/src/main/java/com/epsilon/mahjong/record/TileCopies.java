package com.epsilon.mahjong.record;

import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 表記（Face）しか分からない牌に、1 局の中で物理 id を割り当てる。赤の 5 には、その種類の番号 0 の牌を割り当てる。ほかの牌には、
 * まだ使っていない赤でない牌を番号の小さい順に割り当てる。割り当てを外した物理 id は、また割り当てられる。
 *
 * <p>席ごとの手牌を物理 id で追う。手から出す牌は、同じ表記の牌から番号の小さい順に選ぶ。手出しではツモった牌以外を先に選び（無ければ
 * ツモった牌）、ツモ切りではツモった牌を選ぶ。その表記の牌が手に無ければ IllegalArgumentException を投げる。同時に使えるのは
 * 1 つのスレッドだけ。
 */
public final class TileCopies {

  private final TileSet set;

  /** 物理 id ごとに、この局で割り当て済みか。 */
  private final boolean[] used = new boolean[Tiles.PHYSICAL_IDS];

  /** 席ごと・物理 id ごとに、その牌が手にあるか。 */
  private final boolean[][] held;

  /** 席ごとに、最後にツモった牌。その席が打牌するか、その牌を面子に使うか、割り当てを外すと {@code TurnEvent.NONE} に戻る。 */
  private final int[] drawn;

  public TileCopies(Variant variant) {
    set = variant.tiles();
    held = new boolean[variant.seats()][Tiles.PHYSICAL_IDS];
    drawn = new int[variant.seats()];
    clear();
  }

  /** 局の始めに戻し、どの牌も割り当てていない状態にする。 */
  public void clear() {
    Arrays.fill(used, false);
    for (boolean[] hand : held) {
      Arrays.fill(hand, false);
    }
    Arrays.fill(drawn, TurnEvent.NONE);
  }

  /** 配牌の 1 枚に物理 id を割り当て、その席の手に入れて返す。 */
  public int deal(int seat, Face face) {
    int tile = allocate(face);
    held[seat][tile] = true;
    return tile;
  }

  /** ツモった牌（嶺上牌・補充の牌を含む）に物理 id を割り当て、その席の手に入れて返す。 */
  public int draw(int seat, Face face) {
    int tile = deal(seat, face);
    drawn[seat] = tile;
    return tile;
  }

  /** 手牌に入らない牌（ドラ表示牌・裏ドラ表示牌）に物理 id を割り当てて返す。 */
  public int reveal(Face face) {
    return allocate(face);
  }

  /** 打つ牌の物理 id を選び、手から除いて返す。手出しでも、同じ表記の牌がツモった牌しか無ければ、ツモった牌を選ぶ。 */
  public int discard(int seat, Face face, boolean tsumogiri) {
    int tile;
    if (tsumogiri) {
      tile = drawn[seat] != TurnEvent.NONE && matches(drawn[seat], face) ? drawn[seat] : TurnEvent.NONE;
    } else {
      tile = find(seat, face, drawn[seat]);
      if (tile == TurnEvent.NONE) {
        tile = find(seat, face, TurnEvent.NONE);
      }
    }
    if (tile == TurnEvent.NONE) {
      throw new IllegalArgumentException(
          "Seat " + seat + " holds no " + (tsumogiri ? "drawn " : "") + face.mjai() + " to discard");
    }
    held[seat][tile] = false;
    drawn[seat] = TurnEvent.NONE;
    return tile;
  }

  /** 面子に使う牌の物理 id を選び、手から除いて返す。手にある同じ表記の牌のうち、番号の最も小さい牌を選ぶ。 */
  public int meld(int seat, Face face) {
    int tile = find(seat, face, TurnEvent.NONE);
    if (tile == TurnEvent.NONE) {
      throw new IllegalArgumentException("Seat " + seat + " holds no " + face.mjai() + " to meld");
    }
    held[seat][tile] = false;
    if (drawn[seat] == tile) {
      drawn[seat] = TurnEvent.NONE;
    }
    return tile;
  }

  /** その席が最後にツモった牌。その後に打牌したか、その牌を面子に使ったか、割り当てを外したなら {@code TurnEvent.NONE}。 */
  public int drawn(int seat) {
    return drawn[seat];
  }

  /** その席の手に、その種類の牌が何枚あるか。 */
  public int count(int seat, int kind) {
    int count = 0;
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      count += held[seat][Tiles.physicalId(kind, copy)] ? 1 : 0;
    }
    return count;
  }

  /**
   * 物理 id の割り当てを外して、また割り当てられるようにする。その牌が手にある席からは手から除き、その席が最後にツモった牌なら
   * ツモも取り消す。
   *
   * @param physicalId 割り当てを外す物理 id
   * @throws IllegalArgumentException このルールで使わない牌か、この局で割り当てていない物理 id のとき
   */
  public void release(int physicalId) {
    if (!set.contains(physicalId) || !used[physicalId]) {
      throw new IllegalArgumentException("Tile " + physicalId + " is not assigned in this round");
    }
    used[physicalId] = false;
    for (int seat = 0; seat < held.length; seat++) {
      held[seat][physicalId] = false;
      if (drawn[seat] == physicalId) {
        drawn[seat] = TurnEvent.NONE;
      }
    }
  }

  /**
   * ルールで使う牌のうち、その表記の牌が何枚あるか。赤の 5 は 1 枚、赤がある種類の赤でない 5 は 3 枚、ほかの使う種類は 4 枚、
   * 使わない種類は 0 枚。
   *
   * @param variant ルール
   * @param face 牌の表記
   * @return 枚数
   */
  public static int copies(Variant variant, Face face) {
    TileSet set = variant.tiles();
    int count = 0;
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(face.kind(), copy);
      count += set.contains(tile) && set.red(tile) == face.red() ? 1 : 0;
    }
    return count;
  }

  private int allocate(Face face) {
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(face.kind(), copy);
      if (set.contains(tile) && !used[tile] && set.red(tile) == face.red()) {
        used[tile] = true;
        return tile;
      }
    }
    throw new IllegalArgumentException("No copy of " + face.mjai() + " is left in " + set);
  }

  /**
   * 席の手にある、その表記の牌のうち物理 id が最も小さいもの。無ければ {@code TurnEvent.NONE}。
   *
   * @param avoid 選ばない物理 id。{@code TurnEvent.NONE} なら、どの牌も選べる
   */
  private int find(int seat, Face face, int avoid) {
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(face.kind(), copy);
      if (tile != avoid && held[seat][tile] && set.red(tile) == face.red()) {
        return tile;
      }
    }
    return TurnEvent.NONE;
  }

  private boolean matches(int tile, Face face) {
    return Tiles.kind(tile) == face.kind() && set.red(tile) == face.red();
  }
}
