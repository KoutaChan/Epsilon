package com.epsilon.mahjong.score;

import static com.epsilon.mahjong.shape.ShapeFixtures.tile;
import static com.epsilon.mahjong.shape.ShapeFixtures.tiles;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.RelativeSource;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext.Source;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;

/**
 * テストで組み立てる和了。牌は {@code ShapeFixtures} の表記で書く。純手牌は和了牌を除いて書き、面子と合わせて 13 枚相当にする。
 * 副露を足すと、純手牌の形が持つ副露の数も 1 つ増える。既定は南家・東場で、立直・ドラ・赤は無い。鳴いた相手はいつも上家。
 */
final class WinCase {

  private final WinContext context;

  private WinCase(Variant variant, String concealed) {
    context = new WinContext(variant);
    context.setWinds(Tiles.NAN, Tiles.TON);
    for (int kind : tiles(concealed)) {
      context.hand().add(kind);
    }
  }

  static WinCase of(Variant variant, String concealed) {
    return new WinCase(variant, concealed);
  }

  static WinCase four(String concealed) {
    return of(Variant.TENHOU_4P, concealed);
  }

  static WinCase three(String concealed) {
    return of(Variant.TENHOU_3P, concealed);
  }

  /**
   * 順子のいちばん小さい牌を鳴いたチー。
   *
   * @param run 順子の 3 枚の表記
   */
  WinCase chi(String run) {
    int base = tiles(run)[0];
    return meld(Meld.chi(base, base, AkaSource.NONE));
  }

  WinCase pon(String tile) {
    return meld(Meld.pon(tile(tile), RelativeSource.KAMICHA, AkaSource.NONE));
  }

  WinCase daiminkan(String tile) {
    return meld(Meld.daiminkan(tile(tile), RelativeSource.KAMICHA, AkaSource.NONE));
  }

  /** 5 の暗槓は赤を含む。 */
  WinCase ankan(String tile) {
    int kind = tile(tile);
    return meld(Meld.ankan(kind, Tiles.contains(Tiles.FIVES, kind)));
  }

  WinCase kakan(String tile) {
    return meld(Meld.kakan(Meld.pon(tile(tile), RelativeSource.KAMICHA, AkaSource.NONE), false));
  }

  WinCase meld(Meld meld) {
    context.hand().addMeld();
    context.addMeld(meld);
    return this;
  }

  WinCase ron(String tile) {
    return win(tile, Source.RON);
  }

  WinCase tsumo(String tile) {
    return win(tile, Source.TSUMO);
  }

  WinCase win(String tile, Source source) {
    context.setWin(tile(tile), source);
    return this;
  }

  WinCase dealer() {
    context.setWinds(Tiles.TON, context.roundWind());
    return this;
  }

  WinCase riichi() {
    context.setRiichi(RiichiState.RIICHI, false);
    return this;
  }

  WinCase doubleRiichi() {
    context.setRiichi(RiichiState.DOUBLE_RIICHI, false);
    return this;
  }

  WinCase ippatsu() {
    context.setRiichi(context.riichi(), true);
    return this;
  }

  WinCase firstDraw() {
    context.setFirstDraw(true);
    return this;
  }

  WinCase dora(String indicators) {
    for (int kind : tiles(indicators)) {
      context.addIndicator(kind);
    }
    return this;
  }

  WinCase ura(String indicators) {
    for (int kind : tiles(indicators)) {
      context.addUraIndicator(kind);
    }
    return this;
  }

  /** 純手牌（和了牌を含む）のうち、赤にする 5 を種類で指定する。 */
  WinCase red(String fives) {
    long kinds = 0;
    for (int kind : tiles(fives)) {
      kinds |= Tiles.bit(kind);
    }
    context.setRedFives(kinds);
    return this;
  }

  WinCase nuki(int count) {
    context.setNuki(count);
    return this;
  }

  WinContext context() {
    return context;
  }

  /** 採点の結果を返す。役が無ければテストを失敗させる。 */
  WinScore score() {
    WinScore score = new WinScore();
    Assert.assertTrue(new HandScorer().score(context, score), "no yaku in " + context);
    return score;
  }

  /** 採点の結果を書き込み、役があれば true を返す。 */
  boolean scores(WinScore out) {
    return new HandScorer().score(context, out);
  }

  @Override
  public String toString() {
    return context.toString();
  }
}
