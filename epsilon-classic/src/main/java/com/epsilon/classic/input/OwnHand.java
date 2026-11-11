package com.epsilon.classic.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 自分の手（純手牌、面子、純手牌にある赤）。判断の記録から判断の時点の手を読み込み、合法手を選んだ後の手を作る。入力のフィールドと
 * ルールベースの打ち手は、この作り方を共有する。メモリを確保しない。同時に使えるのは 1 つのスレッドだけ。
 *
 * <p>合法手を選んだ後の手は次のように作る。打牌と立直の打牌は 1 枚を除く（赤を選んだなら赤を除く）。チー・ポンは手から 2 枚（赤を
 * 選んだなら 5 は赤）を除いて面子を足し、鳴いた後に打つ牌を除く。大明槓は 3 枚、暗槓は 4 枚を除いて面子を足す。加槓は 1 枚を除いて、
 * ポンを加槓の面子に替える。槓で除く牌は、手にその種類の赤があれば赤から除く。ほかの手では手を変えない。
 *
 * <p>作る面子の赤の出どころは次のように決める。チー・ポンは、鳴いた牌（直前の打牌）が赤なら鳴いた牌、そうでなく手の赤を使うなら
 * 手の牌。大明槓は、鳴いた牌が赤なら鳴いた牌、そうでなく赤のある 5 なら手の牌。暗槓は、赤のある 5 なら手の牌。加槓は、足す牌が手の
 * 赤なら足した牌、そうでなければポンの出どころを引き継ぐ。鳴いた相手は、直前のイベントの席。
 */
public final class OwnHand {

  private final int seats;
  private final TileSet set;
  private final HandShape shape = new HandShape();
  private final Meld[] melds = new Meld[Variant.MAX_MELDS];
  private int meldCount;
  private long reds;

  public OwnHand(Variant variant) {
    seats = variant.seats();
    set = variant.tiles();
  }

  /**
   * 判断の時点の自分の手を読み込む。
   *
   * @param tiles 判断の記録から作った枚数（純手牌と赤を読む）
   * @param facts 同じ記録から作った席ごとの情報（自分の面子を読む）
   */
  public void load(TileCounts tiles, SeatFacts facts) {
    shape.clear();
    for (long kinds = set.kinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      for (int copy = tiles.hand(kind); copy > 0; copy--) {
        shape.add(kind);
      }
    }
    meldCount = 0;
    for (int index = 0; index < facts.meldCount(0); index++) {
      addMeld(facts.meld(0, index));
    }
    reds = tiles.handRed();
  }

  /**
   * 合法手が作る面子。
   *
   * @param o 判断の記録（鳴いた牌が赤かと、鳴いた相手を読む）
   * @param legal この手の時点の合法手
   * @return 作る面子。チー・ポン・大明槓・暗槓・加槓のほかは null
   * @throws IllegalStateException 加槓の元のポンが無いとき
   */
  public Meld meldOf(Observation o, Action legal) {
    int kind = legal.tile();
    boolean five = Tiles.contains(set.redKinds(), kind);
    return switch (legal.type()) {
      case CHI -> Meld.chi(legal.base(), kind, calledAka(o, legal.selection().red()));
      case PON -> Meld.pon(kind, source(o), calledAka(o, legal.selection().red()));
      case DAIMINKAN -> Meld.daiminkan(kind, source(o), calledAka(o, five));
      case ANKAN -> Meld.ankan(kind, five);
      case KAKAN -> Meld.kakan(pon(kind), Tiles.contains(reds, kind));
      default -> null;
    };
  }

  /**
   * この手を、合法手を選んだ後の手に書き換える。
   *
   * @param current 判断の時点の手（この手とは別のもの）
   * @param legal 判断の時点の合法手
   * @param meld 合法手が作る面子（判断の時点の手で {@link #meldOf} が返す値。作らない手は null）
   * @param discard チー・ポンの後に打つ牌（ほかの手は読まない）
   * @throws IllegalStateException 除く牌、除く赤、加槓の元のポンのどれかが手に無いとき
   */
  public void play(OwnHand current, Action legal, Meld meld, Action discard) {
    shape.copyFrom(current.shape);
    System.arraycopy(current.melds, 0, melds, 0, current.meldCount);
    meldCount = current.meldCount;
    reds = current.reds;
    int kind = legal.tile();
    boolean red = legal.selection().red();
    boolean redHeld = Tiles.contains(current.reds, kind);
    switch (legal.type()) {
      case DAHAI, RIICHI_DAHAI -> remove(kind, red);
      case CHI -> {
        for (int run = legal.base(); run < legal.base() + ClassicVocabulary.RUN; run++) {
          if (run != kind) {
            remove(run, red && Tiles.contains(Tiles.FIVES, run));
          }
        }
        addMeld(meld);
        remove(discard.tile(), discard.selection().red());
      }
      case PON -> {
        remove(kind, red);
        remove(kind, false);
        addMeld(meld);
        remove(discard.tile(), discard.selection().red());
      }
      case DAIMINKAN, ANKAN -> {
        int taken = meld.size() - (legal.type() == Action.Type.DAIMINKAN ? 1 : 0);
        remove(kind, redHeld);
        for (int copy = 1; copy < taken; copy++) {
          remove(kind, false);
        }
        addMeld(meld);
      }
      case KAKAN -> {
        remove(kind, redHeld);
        melds[ponIndex(kind)] = meld;
      }
      default -> {}
    }
  }

  /** 純手牌と面子の数。この手を次に書き換えるまで有効。 */
  public HandShape shape() {
    return shape;
  }

  /** 純手牌にあるその種類の枚数。 */
  public int count(int kind) {
    return shape.count(kind);
  }

  /** 純手牌の枚数。 */
  public int concealed() {
    return shape.tiles();
  }

  /** 面子（副露と暗槓）の数。 */
  public int meldCount() {
    return meldCount;
  }

  /** 面子を成立した順に並べたときの、その位置の面子。加槓したポンは、加槓の面子になっている。 */
  public Meld meld(int index) {
    return melds[index];
  }

  /** 最後に足した面子。 */
  public Meld lastMeld() {
    return melds[meldCount - 1];
  }

  /** 純手牌にある赤の 5 の種類。 */
  public long reds() {
    return reds;
  }

  /** 門前を崩す面子が無ければ true。 */
  public boolean menzen() {
    for (int index = 0; index < meldCount; index++) {
      if (!melds[index].type().concealed()) {
        return false;
      }
    }
    return true;
  }

  /**
   * 純手牌と面子にあるドラの数。牌ごとに、ドラ表示牌から決まる重みを足す。
   *
   * @param tiles 判断の記録から作った枚数（ドラの重みを読む）
   */
  public int dora(TileCounts tiles) {
    int dora = 0;
    for (long kinds = shape.kinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      dora += shape.count(kind) * tiles.dora(kind);
    }
    for (int index = 0; index < meldCount; index++) {
      Meld meld = melds[index];
      for (int tile = 0; tile < meld.size(); tile++) {
        dora += tiles.dora(meld.tile(tile));
      }
    }
    return dora;
  }

  /** 純手牌と面子にある赤の枚数。 */
  public int aka() {
    int aka = Long.bitCount(reds);
    for (int index = 0; index < meldCount; index++) {
      if (melds[index].red()) {
        aka++;
      }
    }
    return aka;
  }

  /**
   * 渡した種類のうち、自分から見えない牌がまだ残っている種類の数。
   *
   * @param tiles 判断の記録から作った枚数
   * @param kinds 種類の集合
   */
  public static int liveKinds(TileCounts tiles, long kinds) {
    int live = 0;
    for (long rest = kinds; rest != 0; rest &= rest - 1) {
      if (tiles.unseen(Long.numberOfTrailingZeros(rest)) > 0) {
        live++;
      }
    }
    return live;
  }

  /**
   * 純手牌から 1 枚除く。
   *
   * @param red 赤の 5 を除くか
   */
  private void remove(int kind, boolean red) {
    shape.remove(kind);
    if (red) {
      if (!Tiles.contains(reds, kind)) {
        throw new IllegalStateException("No red copy of kind " + kind + " in the hand");
      }
      reds &= ~Tiles.bit(kind);
    }
  }

  /** 面子を足す。面子に使う純手牌の牌は、呼ぶ前に除いておく。 */
  private void addMeld(Meld meld) {
    shape.addMeld();
    melds[meldCount++] = meld;
  }

  /** その種類のポン。 */
  private Meld pon(int kind) {
    return melds[ponIndex(kind)];
  }

  /** その種類のポンが面子の何番目か。無ければ IllegalStateException を投げる。 */
  private int ponIndex(int kind) {
    for (int index = 0; index < meldCount; index++) {
      if (melds[index].type() == Meld.Type.PON && melds[index].base() == kind) {
        return index;
      }
    }
    throw new IllegalStateException("No pon of kind " + kind + " for an added kan");
  }

  /** 直前に打牌した席から鳴いたときの、鳴いた相手の向き。 */
  private Meld.RelativeSource source(Observation o) {
    return Meld.RelativeSource.of(0, o.lastEventSeat(), seats);
  }

  /** 鳴いた牌が赤なら鳴いた牌、そうでなく手の赤を使うなら手の牌。 */
  private static Meld.AkaSource calledAka(Observation o, boolean handRed) {
    if (o.lastEventAka()) {
      return Meld.AkaSource.CALLED_TILE;
    }
    return handRed ? Meld.AkaSource.CONSUMED_HAND_TILE : Meld.AkaSource.NONE;
  }
}
