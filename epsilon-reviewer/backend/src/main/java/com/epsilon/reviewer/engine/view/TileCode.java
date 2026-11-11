package com.epsilon.reviewer.engine.view;

import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.HashMap;
import java.util.Map;

/**
 * mjai の表記で書いた牌の記号。萬子・筒子・索子は数字に m・p・s を付け、赤の 5 は後ろに r を付ける。字牌は E S W N P F C。
 * 読むのはこの 37 種類だけで、雀魂の表記（0p・1z など）は受け付けない。局面・候補・副露に書く記号は {@link Face} が持つ文字列を使い回し、新しく作らない。
 */
public final class TileCode {
  /** 記号から、手牌に並べるときの順番への表。 */
  private static final Map<String, Integer> ORDER = new HashMap<>();

  /** 記号から牌の種類と赤（{@link Face}）への表。 */
  private static final Map<String, Face> FACES = new HashMap<>();

  static {
    for (Face face : Face.values()) {
      ORDER.put(face.mjai(), face.kind() * 2 + (face.red() ? 0 : 1));
      FACES.put(face.mjai(), face);
    }
  }

  private TileCode() {}

  /** 牌の種類と赤かどうかから記号を返す。赤になるのは 5 だけ。 */
  public static String of(int kind, boolean red) {
    return Face.of(kind, red).mjai();
  }

  /** 物理牌の記号を返す。赤かどうかは、渡した牌の集合で決まる。 */
  public static String of(TileSet tiles, int physicalId) {
    return of(Tiles.kind(physicalId), tiles.red(physicalId));
  }

  /** 37 種類の記号のどれかなら true。 */
  public static boolean isValid(String code) {
    return code != null && FACES.containsKey(code);
  }

  /** @throws IllegalArgumentException 37 種類の記号のどれでもないとき */
  public static Face face(String code) {
    Face face = code == null ? null : FACES.get(code);
    if (face == null) throw new IllegalArgumentException("Invalid tile code: " + code);
    return face;
  }

  public static int kind(String code) {
    return face(code).kind();
  }

  public static boolean red(String code) {
    return face(code).red();
  }

  /** 手牌に並べるときの順番（protocol/result.ts の tileOrder と同じ）。牌の種類の順で、赤の 5 は同じ種類の 5 の前に来る。 */
  public static int order(String code) {
    Integer order = code == null ? null : ORDER.get(code);
    if (order == null) throw new IllegalArgumentException("Invalid tile code: " + code);
    return order;
  }
}
