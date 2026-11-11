package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * 手で置いた牌で 1 局分の牌山を作り、牌譜の局としてエンジンで始めるテスト用のヘルパー（ほかのモジュールのテストも使う）。配牌、
 * 山のツモ（局全体のツモの順）、嶺上と北抜きの補充、表示牌、裏ドラ表示牌を、牌山の位置の式で決まる所に置く。残りの位置は、乱数の種で
 * 決まる並びで埋める。牌譜の局は席別ツモ列を使わないので、山のツモは位置の順に、嶺上と補充は王牌の嶺上の位置から読む。
 *
 * <p>牌は数字の後に色の文字（m・p・s・z）を付けて書き、0 は赤の 5。物理牌はまだ置いていないものから選び、赤と書かない牌には
 * 赤でないものを使う。牌を置くメソッドは、同じ位置に 2 度置くときと、物理牌が残っていないときに IllegalArgumentException を投げる。
 */
public final class Scenes {

  /** 残りの位置を埋める並びを決める、既定の乱数の種。 */
  public static final long FILL_SEED = 17;

  private final RoundStart start;
  private final Variant variant;
  private final TileSet set;

  /** 位置ごとの、置いた物理牌の id。置いていない位置は {@code Wall.UNKNOWN}。 */
  private final int[] wall;

  private final boolean[] used = new boolean[Tiles.PHYSICAL_IDS];

  /**
   * まだ何も置いていない牌山を作る。
   *
   * @param start 局の始まりの卓の状態
   */
  public Scenes(RoundStart start) {
    this.start = Objects.requireNonNull(start, "start");
    variant = start.variant();
    set = variant.tiles();
    wall = new int[set.size()];
    Arrays.fill(wall, Wall.UNKNOWN);
  }

  /** 東 1 局で、全員の持ち点が配給原点の局。 */
  public static Scenes first(Variant variant) {
    return new Scenes(RoundStart.first(variant));
  }

  /**
   * 局の始まりの卓を値で指定した局。
   *
   * @param variant ルール
   * @param round 局の通し番号
   * @param honba 本場
   * @param kyotaku 供託の本数
   * @param scores 持ち点（絶対席の順）
   */
  public static Scenes at(Variant variant, int round, int honba, int kyotaku, int... scores) {
    return new Scenes(new RoundStart(variant, round, honba, kyotaku, scores));
  }

  /** 局の始まりの卓の状態。 */
  public RoundStart roundStart() {
    return start;
  }

  public Variant variant() {
    return variant;
  }

  /**
   * 席の配牌を置く。
   *
   * @param seat 絶対席
   * @param tiles 13 枚の表記
   */
  public Scenes hand(int seat, String tiles) {
    Face[] faces = faces(tiles);
    if (faces.length != Variant.HAND) {
      throw new IllegalArgumentException("A deal is " + Variant.HAND + " tiles: " + tiles);
    }
    int offset = Math.floorMod(seat - variant.dealer(start.round()), variant.seats());
    for (int index = 0; index < faces.length; index++) {
      place(Wall.dealtPosition(variant, offset, index), faces[index]);
    }
    return this;
  }

  /** 局の最初から、山のツモを局全体のツモの順に置く（鳴きがあれば、その分だけ手番の順とずれる）。 */
  public Scenes draws(String tiles) {
    return drawAt(0, tiles);
  }

  /**
   * 途中からの山のツモを、局全体のツモの順に置く。
   *
   * @param k 最初の牌を置くツモの番号（0 始まり）
   * @param tiles 続けて置く牌の表記
   */
  public Scenes drawAt(int k, String tiles) {
    Face[] faces = faces(tiles);
    for (int index = 0; index < faces.length; index++) {
      place(Wall.livePosition(variant, k + index), faces[index]);
    }
    return this;
  }

  /** 嶺上と北抜きの補充のツモを、1 回目から順に置く。 */
  public Scenes rinshan(String tiles) {
    Face[] faces = faces(tiles);
    for (int index = 0; index < faces.length; index++) {
      place(Wall.rinshanPosition(variant, index + 1), faces[index]);
    }
    return this;
  }

  /** 最初の表示牌を置く。 */
  public Scenes indicator(String tile) {
    return indicator(0, tile);
  }

  /**
   * 表示牌を置く。
   *
   * @param index 何番目の表示牌か（0 始まり。1 から先は槓ドラの表示牌）
   * @param tile 置く牌の表記
   */
  public Scenes indicator(int index, String tile) {
    place(Wall.indicatorPosition(variant, index), single(tile));
    return this;
  }

  /**
   * 裏ドラ表示牌を置く。
   *
   * @param index 何番目の表示牌に対応する裏ドラ表示牌か（0 始まり）
   * @param tile 置く牌の表記
   */
  public Scenes ura(int index, String tile) {
    place(Wall.uraPosition(variant, index), single(tile));
    return this;
  }

  /** 残りの位置を、既定の乱数の種で決まる並びで埋めた牌山。 */
  public int[] wall() {
    return wall(FILL_SEED);
  }

  /**
   * 置いた牌はそのままに、残りの位置を乱数の種で決まる並びで埋めた牌山を返す。呼ぶたびに新しい配列を作り、置いた牌の記録は変えない。
   *
   * @param seed 残りの牌を並べる乱数の種
   * @return 位置ごとの物理 id を並べた牌山（全ての位置が埋まっている）
   */
  public int[] wall(long seed) {
    List<Integer> rest = new ArrayList<>();
    for (int index = 0; index < set.size(); index++) {
      if (!used[set.physicalId(index)]) {
        rest.add(set.physicalId(index));
      }
    }
    Collections.shuffle(rest, new Random(seed));
    int[] filled = wall.clone();
    int next = 0;
    for (int position = 0; position < filled.length; position++) {
      if (filled[position] == Wall.UNKNOWN) {
        filled[position] = rest.get(next++);
      }
    }
    return filled;
  }

  /**
   * 既定の乱数の種で埋めた牌山で、エンジンに牌譜の局を始めさせ、親の最初の手番まで進める。
   *
   * @param engine 局を始めるエンジン（ルールが同じもの）
   * @param ws ルールの作業領域
   * @return 最初の判断の時点
   */
  public Boundary start(Engine engine, RuleWorkspace ws) {
    engine.startRecordedRound(start, wall());
    return engine.advance(ws);
  }

  /** 表記の牌を、書いた順に読む。読めない表記には IllegalArgumentException を投げる。 */
  public static Face[] faces(String text) {
    Face[] faces = new Face[text.length()];
    int size = 0;
    int digits = 0;
    for (int at = 0; at < text.length(); at++) {
      char c = text.charAt(at);
      if (Character.isDigit(c)) {
        digits++;
        continue;
      }
      int first =
          switch (c) {
            case 'm' -> Tiles.M1;
            case 'p' -> Tiles.P1;
            case 's' -> Tiles.S1;
            case 'z' -> Tiles.TON;
            default -> throw new IllegalArgumentException("Unknown suit " + c + " in " + text);
          };
      for (int digit = at - digits; digit < at; digit++) {
        int number = text.charAt(digit) - '0';
        faces[size++] = number == 0 ? Face.of(first + Tiles.FIVE, true) : Face.of(first + number - 1, false);
      }
      digits = 0;
    }
    if (digits > 0) {
      throw new IllegalArgumentException("Numbers without a suit at the end of " + text);
    }
    return Arrays.copyOf(faces, size);
  }

  private static Face single(String tile) {
    Face[] faces = faces(tile);
    if (faces.length != 1) {
      throw new IllegalArgumentException("One tile is expected: " + tile);
    }
    return faces[0];
  }

  /** 表記の 1 枚に当たる、まだ置いていない物理牌を、指定した位置に置く。 */
  private void place(int position, Face face) {
    if (wall[position] != Wall.UNKNOWN) {
      throw new IllegalArgumentException("Wall position " + position + " is set twice");
    }
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int physicalId = Tiles.physicalId(face.kind(), copy);
      if (!used[physicalId] && set.contains(physicalId) && set.red(physicalId) == face.red()) {
        used[physicalId] = true;
        wall[position] = physicalId;
        return;
      }
    }
    throw new IllegalArgumentException("No free copy of " + face.mjai() + " is left in " + set);
  }
}
