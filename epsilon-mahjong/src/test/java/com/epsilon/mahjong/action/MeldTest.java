package com.epsilon.mahjong.action;

import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.RelativeSource;
import com.epsilon.mahjong.action.Meld.Type;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 面子の番号、面子の作り方、鳴き元を確かめる。番号は系譜の記録に残っているので、変えない。 */
public class MeldTest {

  @Test
  public void tableHasTheContractShape() {
    Assert.assertEquals(Meld.IDS, 5 * 34 * 4 * 3 * 4);
  }

  /** 番号 = (((面子の種別 × 34 ＋ 最も小さい牌の種類) × 4 ＋ 鳴き元) × 3 ＋ チーで鳴いた牌の位置) × 4 ＋ 赤 5 の出どころ。 */
  @Test
  public void idsKeepTheirMeaning() {
    Assert.assertEquals(Meld.chi(Tiles.M1, Tiles.M1, AkaSource.NONE).id(), 36);
    Assert.assertEquals(Meld.pon(Tiles.P5, RelativeSource.SHIMOCHA, AkaSource.CALLED_TILE).id(), 2269);
    Assert.assertEquals(Meld.ankan(Tiles.CHUN, false).id(), 6480);
    Assert.assertEquals(Meld.kakan(Meld.pon(Tiles.S5, RelativeSource.TOIMEN, AkaSource.NONE), true).id(), 7611);
  }

  @Test
  public void everyMeldSitsAtItsMixedRadixId() {
    List<Meld> melds = melds();
    for (Meld meld : melds) {
      int position = meld.type() == Type.CHI ? meld.called() - meld.base() : 0;
      int expected =
          (((meld.type().ordinal() * 34 + meld.base()) * 4 + meld.source().ordinal()) * 3 + position) * 4
              + meld.aka().ordinal();
      Assert.assertEquals(meld.id(), expected, meld.toString());
      Assert.assertSame(Meld.of(meld.id()), meld);
    }
    // チーは赤 5 なしが 63、赤 5 入りが 27。暗槓は 5 以外が 31、赤 5 入りの 5 が 3。ポン・大明槓・加槓は、34 種類 × 鳴き元 3 × 3 から
    // 赤の無い 5 の槓子 3 × 3 × 2 を除き、赤 5 の出どころ 2 通り × 3 × 3 × 3 と、加槓で足した赤 5 の 3 × 3 を足す。
    Assert.assertEquals(melds.size(), 63 + 27 + 31 + 3 + 306 - 18 + 54 + 9);
  }

  /** 5 の槓子は同じ種類の 4 枚を全て使うので、どの牌の集合でも赤 5 を含む。そのため赤の無い 5 の槓子は作れない。 */
  @Test
  public void fiveKansAlwaysHoldTheRedFive() {
    for (TileSet set : TileSet.values()) {
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        if (set.copies(kind) > 0 && Tiles.contains(Tiles.FIVES, kind)) {
          Assert.assertTrue(set.red(Tiles.physicalId(kind, Tiles.RED_COPY)), set + " kind " + kind);
        }
      }
    }
    for (int five : new int[] {Tiles.M5, Tiles.P5, Tiles.S5}) {
      Assert.assertThrows(IllegalArgumentException.class, () -> Meld.ankan(five, false));
      for (RelativeSource source : RelativeSource.values()) {
        if (source != RelativeSource.NONE) {
          Assert.assertThrows(IllegalArgumentException.class, () -> Meld.daiminkan(five, source, AkaSource.NONE));
          Meld pon = Meld.pon(five, source, AkaSource.NONE);
          Assert.assertThrows(IllegalArgumentException.class, () -> Meld.kakan(pon, false));
          Assert.assertSame(Meld.kakan(pon, true).aka(), AkaSource.ADDED_KAN_TILE);
        }
      }
    }
  }

  @Test
  public void factoriesRebuildEveryMeld() {
    for (Meld meld : melds()) {
      Meld rebuilt =
          switch (meld.type()) {
            case CHI -> Meld.chi(meld.base(), meld.called(), meld.aka());
            case PON -> Meld.pon(meld.base(), meld.source(), meld.aka());
            case DAIMINKAN -> Meld.daiminkan(meld.base(), meld.source(), meld.aka());
            case ANKAN -> Meld.ankan(meld.base(), meld.red());
            case KAKAN ->
                meld.aka() == AkaSource.ADDED_KAN_TILE
                    ? Meld.kakan(Meld.pon(meld.base(), meld.source(), AkaSource.NONE), true)
                    : Meld.kakan(Meld.pon(meld.base(), meld.source(), meld.aka()), false);
          };
      Assert.assertSame(rebuilt, meld);
    }
  }

  @Test
  public void impossibleMeldsAreRejected() {
    Assert.assertThrows(IllegalArgumentException.class, () -> Meld.chi(Tiles.TON, Tiles.TON, AkaSource.NONE));
    Assert.assertThrows(IllegalArgumentException.class, () -> Meld.chi(Tiles.M8, Tiles.M8, AkaSource.NONE));
    Assert.assertThrows(IllegalArgumentException.class, () -> Meld.chi(Tiles.M1, Tiles.M4, AkaSource.NONE));
    Assert.assertThrows(IllegalArgumentException.class, () -> Meld.chi(Tiles.M1, Tiles.M1, AkaSource.CALLED_TILE));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Meld.chi(Tiles.M3, Tiles.M5, AkaSource.CONSUMED_HAND_TILE));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Meld.pon(Tiles.M1, RelativeSource.NONE, AkaSource.NONE));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Meld.pon(Tiles.M1, RelativeSource.SHIMOCHA, AkaSource.CALLED_TILE));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Meld.pon(Tiles.M5, RelativeSource.SHIMOCHA, AkaSource.ADDED_KAN_TILE));
    Assert.assertThrows(IllegalArgumentException.class, () -> Meld.pon(-1, RelativeSource.SHIMOCHA, AkaSource.NONE));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Meld.kakan(Meld.chi(Tiles.M1, Tiles.M1, AkaSource.NONE), false));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Meld.kakan(Meld.pon(Tiles.M5, RelativeSource.KAMICHA, AkaSource.CALLED_TILE), true));
    Assert.assertThrows(IllegalArgumentException.class, () -> Meld.of(-1));
    Assert.assertThrows(IllegalArgumentException.class, () -> Meld.of(1));
    Assert.assertThrows(IllegalArgumentException.class, () -> Meld.of(Meld.IDS));
  }

  @Test
  public void kakanKeepsTheSourceAndTheRedOfItsPon() {
    Meld kan = Meld.kakan(Meld.pon(Tiles.P5, RelativeSource.KAMICHA, AkaSource.CONSUMED_HAND_TILE), false);
    Assert.assertSame(kan.type(), Type.KAKAN);
    Assert.assertSame(kan.source(), RelativeSource.KAMICHA);
    Assert.assertSame(kan.aka(), AkaSource.CONSUMED_HAND_TILE);
    Assert.assertEquals(kan.called(), Tiles.P5);
    Assert.assertEquals(kan.size(), 4);
    Meld added = Meld.kakan(Meld.pon(Tiles.P5, RelativeSource.KAMICHA, AkaSource.NONE), true);
    Assert.assertSame(added.aka(), AkaSource.ADDED_KAN_TILE);
  }

  @Test
  public void tilesAndKinds() {
    Meld chi = Meld.chi(Tiles.S3, Tiles.S4, AkaSource.CONSUMED_HAND_TILE);
    Assert.assertEquals(chi.size(), 3);
    Assert.assertEquals(chi.tile(0), Tiles.S3);
    Assert.assertEquals(chi.tile(2), Tiles.S5);
    Assert.assertEquals(chi.kinds(), Tiles.range(Tiles.S3, Tiles.S5));
    Assert.assertSame(chi.source(), RelativeSource.KAMICHA);
    Assert.assertTrue(chi.red());
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> chi.tile(3));
    Meld ankan = Meld.ankan(Tiles.M5, true);
    Assert.assertEquals(ankan.size(), 4);
    Assert.assertEquals(ankan.tile(3), Tiles.M5);
    Assert.assertEquals(ankan.kinds(), Tiles.bit(Tiles.M5));
    Assert.assertEquals(ankan.called(), -1);
    Assert.assertSame(ankan.source(), RelativeSource.NONE);
    Assert.assertSame(ankan.aka(), AkaSource.CONSUMED_HAND_TILE);
  }

  @Test
  public void typesDeclareTheirShape() {
    Assert.assertSame(Type.CHI.requires(), RuleFlag.CHI);
    for (Type type : Type.values()) {
      Assert.assertEquals(type.tiles(), type == Type.CHI || type == Type.PON ? 3 : 4, type.name());
      Assert.assertEquals(type.kan(), type.tiles() == 4, type.name());
      Assert.assertEquals(type.concealed(), type == Type.ANKAN, type.name());
      if (type != Type.CHI) {
        Assert.assertNull(type.requires(), type.name());
      }
    }
  }

  /**
   * 鳴き元は、鳴いた席から牌を出した席までの距離（席の数で割った余り）で決まる。逆に、鳴き元から牌を出した席も求められる。
   * 三人打ちに対面は無い。
   */
  @Test
  public void relativeSourcesFollowTheSeatDistance() {
    for (int seats = 3; seats <= 4; seats++) {
      for (int seat = 0; seat < seats; seat++) {
        for (int source = 0; source < seats; source++) {
          int distance = Math.floorMod(source - seat, seats);
          RelativeSource expected =
              distance == 0
                  ? RelativeSource.NONE
                  : distance == 1
                      ? RelativeSource.SHIMOCHA
                      : distance == seats - 1 ? RelativeSource.KAMICHA : RelativeSource.TOIMEN;
          RelativeSource relative = RelativeSource.of(seat, source, seats);
          Assert.assertSame(relative, expected, seats + " seats, " + seat + " from " + source);
          Assert.assertEquals(relative.seat(seat, seats), source);
          Assert.assertFalse(seats == 3 && relative == RelativeSource.TOIMEN);
        }
      }
    }
    Assert.assertSame(RelativeSource.of(0, 2, 4), RelativeSource.TOIMEN);
    Assert.assertSame(RelativeSource.of(0, 2, 3), RelativeSource.KAMICHA);
    Assert.assertThrows(IllegalArgumentException.class, () -> RelativeSource.TOIMEN.seat(0, 3));
    Assert.assertThrows(IllegalArgumentException.class, () -> RelativeSource.of(0, 0, 5));
    Assert.assertThrows(IllegalArgumentException.class, () -> RelativeSource.of(3, 0, 3));
  }

  private static List<Meld> melds() {
    List<Meld> melds = new ArrayList<>();
    for (int id = 0; id < Meld.IDS; id++) {
      try {
        melds.add(Meld.of(id));
      } catch (IllegalArgumentException unused) {
        // 使わない番号。
      }
    }
    return melds;
  }
}
