package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import java.util.Objects;

/**
 * 天鳳の牌の番号と、ツモ・打牌のタグを読み書きする。牌の番号は物理 id と同じ 0〜135 で、赤の 5 は、5 の 4 枚のうち番号が最も小さい
 * 牌（16・52・88）。タグの名前は、席ごとの 1 文字の後に牌の番号が続く（T89 は席 0 が 89 をツモる、E73 は席 1 が 73 を打つ）。
 */
public final class TenhouTiles {

  private static final String SEPARATOR = ",";

  /** ツモか打牌か。席ごとの文字は、先頭の文字に席の番号を足したもの。 */
  public enum Move {
    DRAW('T'),
    DISCARD('D');

    private final char first;

    Move(char first) {
      this.first = first;
    }

    /** タグの名前の先頭の文字から席を読む。この種類の文字でないか、そのルールに無い席なら −1 を返す。 */
    public int seat(char letter, Variant variant) {
      int seat = letter - first;
      return seat >= 0 && seat < variant.seats() ? seat : -1;
    }
  }

  /** ツモか打牌のタグ 1 つ。 */
  public record TileTag(Move move, int seat, int tile) {

    public TileTag {
      Objects.requireNonNull(move, "move");
    }

    /**
     * タグの名前がツモか打牌のタグ（1 文字の後に番号）なら読み、違えば null を返す。牌の番号がそのルールの牌に無ければ
     * IllegalArgumentException を投げる。
     */
    public static TileTag parse(String name, Variant variant) {
      if (name.length() < 2 || !digits(name)) {
        return null;
      }
      for (Move move : Move.values()) {
        int seat = move.seat(name.charAt(0), variant);
        if (seat >= 0) {
          return new TileTag(move, seat, TenhouTiles.tile(Integer.parseInt(name.substring(1)), variant.tiles()));
        }
      }
      return null;
    }

    public String name() {
      return String.valueOf((char) (move.first + seat)) + tile;
    }

    private static boolean digits(String name) {
      for (int i = 1; i < name.length(); i++) {
        if (!Character.isDigit(name.charAt(i))) {
          return false;
        }
      }
      return true;
    }
  }

  private TenhouTiles() {}

  /** 番号をそのまま牌の物理 id として返す。その卓の牌に無ければ IllegalArgumentException を投げる。 */
  public static int tile(int number, TileSet set) {
    if (!set.contains(number)) {
      throw new IllegalArgumentException("Tenhou tile " + number + " is not in " + set);
    }
    return number;
  }

  /** カンマで区切った牌の番号の列を読む。空の文字列なら長さ 0 の配列を返す。 */
  public static int[] tiles(String list, TileSet set) {
    if (list.isEmpty()) {
      return new int[0];
    }
    String[] numbers = list.split(SEPARATOR, -1);
    int[] tiles = new int[numbers.length];
    for (int index = 0; index < numbers.length; index++) {
      tiles[index] = tile(Integer.parseInt(numbers[index].trim()), set);
    }
    return tiles;
  }
}
