package com.epsilon.mahjong.tile;

import java.util.HashMap;
import java.util.Map;

/**
 * 37 通りの牌の表記。先頭の 34 個は種類の番号の順（ordinal = 種類）に並び、続く 3 つが赤 5。
 *
 * <p>mjai の表記は数の後に m・p・s を付け、赤 5 はさらに r を付ける。字牌は E S W N P F C。雀魂の表記は、赤 5 を数 0 で、
 * 字牌を 1z〜7z で書く。
 */
public enum Face {
  M1,
  M2,
  M3,
  M4,
  M5,
  M6,
  M7,
  M8,
  M9,
  P1,
  P2,
  P3,
  P4,
  P5,
  P6,
  P7,
  P8,
  P9,
  S1,
  S2,
  S3,
  S4,
  S5,
  S6,
  S7,
  S8,
  S9,
  TON,
  NAN,
  SHA,
  PEI,
  HAKU,
  HATSU,
  CHUN,
  M5_RED(Tiles.M5),
  P5_RED(Tiles.P5),
  S5_RED(Tiles.S5);

  private static final Face[] VALUES = values();

  /** 種類 → その種類の赤 5。赤の無い種類は null。 */
  private static final Face[] RED_OF_KIND = redOfKind();

  private static final Map<String, Face> BY_TEXT = byText();

  private final int kind;
  private final boolean red;
  private final String mjai;
  private final String majsoul;

  Face() {
    kind = ordinal();
    red = false;
    mjai = mjaiText(kind, false);
    majsoul = majsoulText(kind, false);
  }

  Face(int five) {
    kind = five;
    red = true;
    mjai = mjaiText(five, true);
    majsoul = majsoulText(five, true);
  }

  public int kind() {
    return kind;
  }

  public boolean red() {
    return red;
  }

  public String mjai() {
    return mjai;
  }

  /** 種類と赤かどうかから表記を返す。赤があるのは 5m・5p・5s だけで、ほかの種類の赤なら IllegalArgumentException を投げる。 */
  public static Face of(int kind, boolean red) {
    if (kind < 0 || kind >= Tiles.KINDS) {
      throw new IllegalArgumentException("Tile kind out of range: " + kind);
    }
    Face face = red ? RED_OF_KIND[kind] : VALUES[kind];
    if (face == null) {
      throw new IllegalArgumentException("Only fives can be red: " + VALUES[kind]);
    }
    return face;
  }

  /** mjai か雀魂の表記を読む。知らない表記なら IllegalArgumentException を投げる。 */
  public static Face parse(String text) {
    Face face = text == null ? null : BY_TEXT.get(text);
    if (face == null) {
      throw new IllegalArgumentException("Unknown tile notation: " + text);
    }
    return face;
  }

  private static String mjaiText(int kind, boolean red) {
    Tiles.Group group = Tiles.Group.of(kind);
    if (!group.suited()) {
      return mjaiHonor(kind);
    }
    return (Tiles.number(kind) + 1) + suitLetter(group) + (red ? "r" : "");
  }

  /** 雀魂の表記。字牌は東から中まで、種類の番号の順に 1z〜7z と書く。 */
  private static String majsoulText(int kind, boolean red) {
    Tiles.Group group = Tiles.Group.of(kind);
    if (!group.suited()) {
      return (kind - Tiles.TON + 1) + "z";
    }
    return (red ? 0 : Tiles.number(kind) + 1) + suitLetter(group);
  }

  /** 色を表す文字。mjai と雀魂で共通。 */
  private static String suitLetter(Tiles.Group suit) {
    return switch (suit) {
      case MAN -> "m";
      case PIN -> "p";
      case SOU -> "s";
      case WIND, DRAGON -> throw new IllegalArgumentException("Not a suit: " + suit);
    };
  }

  private static String mjaiHonor(int kind) {
    return switch (kind) {
      case Tiles.TON -> "E";
      case Tiles.NAN -> "S";
      case Tiles.SHA -> "W";
      case Tiles.PEI -> "N";
      case Tiles.HAKU -> "P";
      case Tiles.HATSU -> "F";
      case Tiles.CHUN -> "C";
      default -> throw new IllegalArgumentException("Not an honor kind: " + kind);
    };
  }

  private static Face[] redOfKind() {
    Face[] faces = new Face[Tiles.KINDS];
    for (Face face : VALUES) {
      if (face.red) {
        faces[face.kind] = face;
      }
    }
    return faces;
  }

  private static Map<String, Face> byText() {
    Map<String, Face> faces = new HashMap<>();
    for (Face face : VALUES) {
      for (String text : new String[] {face.mjai, face.majsoul}) {
        Face previous = faces.put(text, face);
        if (previous != null && previous != face) {
          throw new IllegalStateException("Tile notation " + text + " names both " + previous + " and " + face);
        }
      }
    }
    return faces;
  }
}
