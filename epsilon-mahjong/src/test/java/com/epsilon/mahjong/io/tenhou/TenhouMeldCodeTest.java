package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 天鳳の面子と北抜きのコード（m の値）を読み書きするテスト。 */
public class TenhouMeldCodeTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final TileSet SET = VARIANT.tiles();

  /** 全ての面子と席で物理牌の組み合わせを全て試す。赤の由来が合う組み合わせは書いて読むと元に戻り、合わない組み合わせは書かない。 */
  @Test
  public void everyMeldWithEveryCopyRoundTrips() {
    int melds = 0;
    for (int id = 0; id < Meld.IDS; id++) {
      Meld meld = meld(id);
      if (meld == null) {
        continue;
      }
      int fitting = 0;
      for (TenhouMeldCode tiles : tilings(meld)) {
        for (int seat = 0; seat < VARIANT.seats(); seat++) {
          int caller = seat;
          if (aka(tiles) == meld.aka()) {
            Assert.assertEquals(
                TenhouMeldCode.decode(tiles.encode(VARIANT, caller), VARIANT, caller), tiles, tiles.toString());
            fitting++;
          } else {
            Assert.assertThrows(IllegalArgumentException.class, () -> tiles.encode(VARIANT, caller));
          }
        }
      }
      Assert.assertTrue(fitting > 0, meld.toString());
      melds++;
    }
    Assert.assertTrue(melds > 400, melds + " melds");
  }

  /** 自己対局の半荘（四人打ちと三人打ち）を書いた XML の面子と北抜きのコードは、どれも鳴いた席で読んで書き直すと同じコードになる。 */
  @Test
  public void everyCodeOfSelfPlayGamesRoundTrips() {
    Pattern tag = Pattern.compile("<N who=\"(\\d)\" m=\"(\\d+)\"");
    int melds = 0;
    int nukis = 0;
    for (Variant variant : Variant.values()) {
      for (int seed = 0; seed < 20; seed++) {
        String id = variant + " self-" + seed;
        Matcher matcher = tag.matcher(TenhouXml.write(RecordedGames.play(variant, seed, id)));
        while (matcher.find()) {
          int seat = Integer.parseInt(matcher.group(1));
          int code = Integer.parseInt(matcher.group(2));
          String where = id + " who=" + seat + " m=" + code;
          if (TenhouMeldCode.nuki(code)) {
            Assert.assertEquals(TenhouMeldCode.encodeNuki(TenhouMeldCode.nukiTile(code, variant)), code, where);
            nukis++;
          } else {
            Assert.assertEquals(TenhouMeldCode.decode(code, variant, seat).encode(variant, seat), code, where);
            melds++;
          }
        }
      }
    }
    Assert.assertTrue(melds > 100, melds + " meld codes");
    Assert.assertTrue(nukis > 100, nukis + " nuki codes");
  }

  /**
   * 三人打ちの鳴き元のフィールドは、空き席も数えた 4 席での距離。席 1 が上家（席 0）から鳴けば 3、席 2 が下家（席 0）から鳴けば 2 に
   * なる。空き席を指す距離は読まない。
   */
  @Test
  public void threePlayerSourcesCountTheEmptySeat() {
    Variant sanma = Variant.TENHOU_3P;
    int[][] expected = {
      // 鳴いた席、鳴き元の席、鳴き元のフィールドの値
      {0, 1, 1}, {0, 2, 2}, {1, 2, 1}, {1, 0, 3}, {2, 0, 2}, {2, 1, 3}
    };
    for (int[] call : expected) {
      Meld.RelativeSource source = Meld.RelativeSource.of(call[0], call[1], sanma.seats());
      TenhouMeldCode pon =
          new TenhouMeldCode(Meld.pon(Tiles.S9, source, Meld.AkaSource.NONE), 104, new int[] {105, 106});
      int code = pon.encode(sanma, call[0]);
      Assert.assertEquals(code & 3, call[2], Arrays.toString(call));
      Assert.assertEquals(TenhouMeldCode.decode(code, sanma, call[0]), pon, Arrays.toString(call));
    }
    Meld kamicha = Meld.pon(Tiles.S9, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE);
    int distanceThree = new TenhouMeldCode(kamicha, 104, new int[] {105, 106}).encode(sanma, 1);
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.decode(distanceThree, sanma, 0));
  }

  /** 北抜きのコードは、抜いた北の物理 id を 8 bit 左にずらして 0x20 を足した値。北抜きの無い卓や北でない牌は読み書きしない。 */
  @Test
  public void nukiCodesCarryTheNorth() {
    Variant sanma = Variant.TENHOU_3P;
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int north = Tiles.physicalId(Tiles.PEI, copy);
      int code = TenhouMeldCode.encodeNuki(north);
      Assert.assertEquals(code, north << 8 | 0x20);
      Assert.assertTrue(TenhouMeldCode.nuki(code));
      Assert.assertEquals(TenhouMeldCode.nukiTile(code, sanma), north);
      Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.decode(code, sanma, 0));
      Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.nukiTile(code, VARIANT));
    }
    Assert.assertEquals(TenhouMeldCode.nukiTile(31008, sanma), 121);
    Assert.assertFalse(TenhouMeldCode.nuki(16384));
    Assert.assertFalse(TenhouMeldCode.nuki(15543));
    Assert.assertFalse(TenhouMeldCode.nuki(-224));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.nukiTile(0x20 | 32 << 8, sanma));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.nukiTile(31008 | 1 << 6, sanma));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.encodeNuki(119));
  }

  /** 天鳳の牌譜に現れるコードを読む（期待する面子は直前の打牌と照合して決めた）。四人打ちの鳴き元の向きは鳴いた席に依らない。 */
  @Test
  public void codesFromTheLogsReadAsTheirMelds() {
    Assert.assertEquals(
        TenhouMeldCode.decode(15543, VARIANT, 0),
        new TenhouMeldCode(Meld.chi(Tiles.M6, Tiles.M6, Meld.AkaSource.NONE), 22, new int[] {25, 29}));
    Assert.assertEquals(
        TenhouMeldCode.decode(43562, VARIANT, 1),
        new TenhouMeldCode(
            Meld.pon(Tiles.NAN, Meld.RelativeSource.TOIMEN, Meld.AkaSource.NONE), 114, new int[] {112, 115}));
    Assert.assertEquals(
        TenhouMeldCode.decode(4098, VARIANT, 2),
        new TenhouMeldCode(
            Meld.daiminkan(Tiles.M5, Meld.RelativeSource.TOIMEN, Meld.AkaSource.CALLED_TILE),
            16,
            new int[] {17, 18, 19}));
    Assert.assertEquals(
        TenhouMeldCode.decode(16384, VARIANT, 3),
        new TenhouMeldCode(Meld.ankan(Tiles.P8, false), -1, new int[] {64, 65, 66, 67}));
    Meld pon = Meld.pon(Tiles.CHUN, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE);
    Assert.assertEquals(TenhouMeldCode.decode(51211, VARIANT, 0), new TenhouMeldCode(pon, 134, new int[] {133, 135}));
    Assert.assertEquals(
        TenhouMeldCode.decode(51219, VARIANT, 0), new TenhouMeldCode(Meld.kakan(pon, false), 134, new int[] {132}));
  }

  /** コードの形をしていない値と、チーの無い三人打ちの卓でのチー（上家からの 1p2p3p）は読まない。 */
  @Test
  public void valuesThatAreNotCodesAreRejected() {
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.decode(1 << 16, VARIANT, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.decode(-1, VARIANT, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.decode(15543 & ~3 | 1, VARIANT, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.decode(16384 | 1 << 5, VARIANT, 0));
    TenhouMeldCode chi =
        new TenhouMeldCode(Meld.chi(Tiles.P1, Tiles.P1, Meld.AkaSource.NONE), 36, new int[] {40, 44});
    int code = chi.encode(VARIANT, 0);
    Assert.assertEquals(TenhouMeldCode.decode(code, VARIANT, 0), chi);
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouMeldCode.decode(code, Variant.TENHOU_3P, 1));
  }

  private static Meld meld(int id) {
    try {
      return Meld.of(id);
    } catch (IllegalArgumentException unused) {
      return null;
    }
  }

  /** 面子の種別・牌の種類・鳴き元はそのままに、物理牌の選び方を全て並べる（赤の由来が合わないものも含む）。 */
  private static List<TenhouMeldCode> tilings(Meld meld) {
    List<TenhouMeldCode> tilings = new ArrayList<>();
    int kind = meld.base();
    switch (meld.type()) {
      case CHI -> {
        int[] others = new int[2];
        int size = 0;
        for (int index = 0; index < meld.size(); index++) {
          if (meld.tile(index) != meld.called()) {
            others[size++] = meld.tile(index);
          }
        }
        for (int called = 0; called < Tiles.COPIES; called++) {
          for (int first = 0; first < Tiles.COPIES; first++) {
            for (int second = 0; second < Tiles.COPIES; second++) {
              tilings.add(
                  new TenhouMeldCode(
                      meld,
                      Tiles.physicalId(meld.called(), called),
                      new int[] {Tiles.physicalId(others[0], first), Tiles.physicalId(others[1], second)}));
            }
          }
        }
      }
      case PON -> {
        for (int called = 0; called < Tiles.COPIES; called++) {
          for (int left = 0; left < Tiles.COPIES; left++) {
            if (left != called) {
              tilings.add(new TenhouMeldCode(meld, Tiles.physicalId(kind, called), others(kind, called, left)));
            }
          }
        }
      }
      case DAIMINKAN -> {
        for (int called = 0; called < Tiles.COPIES; called++) {
          tilings.add(new TenhouMeldCode(meld, Tiles.physicalId(kind, called), others(kind, called, -1)));
        }
      }
      case ANKAN -> tilings.add(new TenhouMeldCode(meld, -1, others(kind, -1, -1)));
      case KAKAN -> {
        for (int added = 0; added < Tiles.COPIES; added++) {
          for (int called = 0; called < Tiles.COPIES; called++) {
            if (called != added) {
              tilings.add(
                  new TenhouMeldCode(meld, Tiles.physicalId(kind, called), new int[] {Tiles.physicalId(kind, added)}));
            }
          }
        }
      }
    }
    return tilings;
  }

  /**
   * 1 種類の 4 枚から、指定した牌を除いた物理牌を返す。
   *
   * @param kind 牌の種類
   * @param first 除く牌の、同じ種類の中での番号（−1 なら除かない）
   * @param second 除くもう 1 枚の牌の、同じ種類の中での番号（−1 なら除かない）
   */
  private static int[] others(int kind, int first, int second) {
    List<Integer> tiles = new ArrayList<>();
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      if (copy != first && copy != second) {
        tiles.add(Tiles.physicalId(kind, copy));
      }
    }
    return tiles.stream().mapToInt(Integer::intValue).toArray();
  }

  /** 物理牌の組が表す赤の由来。 */
  private static Meld.AkaSource aka(TenhouMeldCode tiles) {
    int[] hand = tiles.hand();
    if (tiles.meld().type() == Meld.Type.KAKAN) {
      if (SET.red(hand[0])) {
        return Meld.AkaSource.ADDED_KAN_TILE;
      }
      hand = others(tiles.meld().base(), Tiles.copy(hand[0]), Tiles.copy(tiles.called()));
    }
    if (tiles.called() >= 0 && SET.red(tiles.called())) {
      return Meld.AkaSource.CALLED_TILE;
    }
    for (int tile : hand) {
      if (SET.red(tile)) {
        return Meld.AkaSource.CONSUMED_HAND_TILE;
      }
    }
    return Meld.AkaSource.NONE;
  }
}
