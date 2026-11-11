package com.epsilon.prism.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Objects;

/**
 * 手を打った後の自分の手（純手牌の形・赤・面子）。判断の時点の手（{@link #loadCurrent}）を写し、選んだ手と鳴いた後の打牌を反映して作る。
 * 形を調べるときは {@link #shape} を使い、採点するときは {@link #writeTo} で採点の条件に手を書く。メモリを割り当てない。
 *
 * <p>手の動かし方はエンジンと同じにする。打牌は、赤を選んだときか赤しか残っていないときに赤を捨てる。チー・ポンは、赤を選んだときか普通の牌が
 * 足りないときに手の赤を使う。大明槓・暗槓は手の赤を先に使う。加槓は、手に赤があれば赤を足す。北抜きは手の北を 1 枚抜く。面子の鳴き元と赤の
 * 出どころは、直前の打牌（判断の記録の直前のイベント）から決める。和了・九種九牌・見送りは手を変えない。面子は鳴いた順に持ち、加槓はポンを
 * 置き換える。抜いた北は面子に数えず、枚数だけを持つ。
 *
 * <p>面子の牌の表ドラは、判断の時点の手で 1 回数え、その後は面子を足すときにその面子の分だけ足す（手を動かすたびに数え直さない）。
 */
public final class AfterHand {

  private final HandShape shape = new HandShape();
  private final Meld[] melds = new Meld[Variant.MAX_MELDS];
  private int meldCount;
  private int seats;
  private TileCounts tiles;

  /** 純手牌にある赤の 5 の種類。 */
  private long concealedRed;

  /** 純手牌と面子にある赤の 5 の種類。 */
  private long ownedRed;

  private int meldDora;
  private boolean menzen;

  /** 抜いた北の数。 */
  private int nuki;

  /**
   * 判断の時点の自分の手にする。
   *
   * @param o 判断の記録（席の数を読む）
   * @param tiles 種類ごとの枚数（手の枚数・赤・ドラの数を読む。この判断の間、この手とここから写した手が参照し続ける）
   * @param seatFacts 席ごとの公開情報（自分の今の面子を読む）
   */
  public void loadCurrent(Observation o, TileCounts tiles, SeatFacts seatFacts) {
    this.tiles = tiles;
    seats = o.variant().seats();
    shape.clear();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      for (int copy = tiles.hand(kind); copy > 0; copy--) {
        shape.add(kind);
      }
    }
    meldCount = 0;
    concealedRed = tiles.handRed();
    ownedRed = concealedRed;
    meldDora = 0;
    menzen = true;
    nuki = o.nuki(0);
    for (int index = 0, count = seatFacts.meldCount(0); index < count; index++) {
      addMeld(seatFacts.meld(0, index));
    }
  }

  /** 別の手の中身をこの手に写す。 */
  public void copyFrom(AfterHand source) {
    shape.copyFrom(source.shape);
    System.arraycopy(source.melds, 0, melds, 0, source.meldCount);
    meldCount = source.meldCount;
    seats = source.seats;
    tiles = source.tiles;
    concealedRed = source.concealedRed;
    ownedRed = source.ownedRed;
    meldDora = source.meldDora;
    menzen = source.menzen;
    nuki = source.nuki;
  }

  /**
   * 合法手を打った後の手にする。チー・ポンの後の打牌は {@link #discard} で別に反映する。
   *
   * @param action 合法手
   * @param o 判断の記録（鳴く牌を出した席・その牌・赤かどうかを、直前のイベントから読む）
   */
  public void apply(Action action, Observation o) {
    int tile = action.tile();
    switch (action.type()) {
      case DAHAI, RIICHI_DAHAI -> discard(action);
      case CHI -> chi(action, o.lastEventTile(), o.lastEventAka());
      case PON -> pon(tile, action.selection().red(), o.lastEventSeat(), o.lastEventAka());
      case DAIMINKAN -> daiminkan(tile, o.lastEventSeat(), o.lastEventAka());
      case ANKAN -> ankan(tile);
      case KAKAN -> kakan(tile);
      case NUKI -> {
        shape.remove(Tiles.PEI);
        nuki++;
      }
      case TSUMO_AGARI, RON_AGARI, KYUSHU_KYUHAI, PASS -> {}
    }
  }

  /** 打牌を反映する。赤を選んだときか、赤しか残っていないときに赤を捨てる。 */
  public void discard(Action discard) {
    int tile = discard.tile();
    if (hasRed(tile) && (discard.selection().red() || plain(tile) == 0)) {
      removeRed(tile);
    } else {
      shape.remove(tile);
    }
  }

  /** 純手牌の形（面子の数を含む）。読むだけにし、書き換えない。 */
  public HandShape shape() {
    return shape;
  }

  /** 面子を 3 枚と数えた手の枚数（純手牌 ＋ 3 × 面子の数）。 */
  public int effectiveTileCount() {
    return shape.size();
  }

  /** 純手牌にあるその種類の牌の枚数。 */
  public int count(int kind) {
    return shape.count(kind);
  }

  /** 純手牌にある種類のマスク。 */
  public long concealedKinds() {
    return shape.kinds();
  }

  public int meldCount() {
    return meldCount;
  }

  public Meld meld(int index) {
    return melds[Objects.checkIndex(index, meldCount)];
  }

  /** 門前か（暗槓のほかに面子がない）。 */
  public boolean isMenzen() {
    return menzen;
  }

  /** 抜いた北の数。 */
  public int nuki() {
    return nuki;
  }

  /** 手・面子・抜いた北にあるドラの数（表ドラ・赤・抜きドラ。裏ドラは数えない）。抜いた北は 1 枚につき、1 と北を指す表示の数を足す。 */
  public int dora() {
    int dora = Long.bitCount(ownedRed) + meldDora + nuki * (1 + tiles.dora(Tiles.PEI));
    for (long rest = shape.kinds() & tiles.doraKinds(); rest != 0L; rest &= rest - 1) {
      int kind = Long.numberOfTrailingZeros(rest);
      dora += shape.count(kind) * tiles.dora(kind);
    }
    return dora;
  }

  /** 採点の条件に、純手牌・面子・純手牌の赤・抜いた北を書く。呼ぶ前に、採点の条件の手と面子を空にしておく。 */
  public void writeTo(WinContext context) {
    context.hand().copyFrom(shape);
    for (int index = 0; index < meldCount; index++) {
      context.addMeld(melds[index]);
    }
    context.setRedFives(concealedRed);
    context.setNuki(nuki);
  }

  private void chi(Action action, int calledTile, boolean calledAka) {
    int base = action.base();
    boolean consumedAka = false;
    for (int tile = base; tile < base + Meld.Type.CHI.tiles(); tile++) {
      if (tile == calledTile) {
        continue;
      }
      if (hasRed(tile) && (action.selection().red() || plain(tile) == 0)) {
        removeRed(tile);
        consumedAka = true;
      } else {
        shape.remove(tile);
      }
    }
    addMeld(Meld.chi(base, calledTile, akaSource(calledAka, consumedAka)));
  }

  private void pon(int tile, boolean usesAka, int source, boolean calledAka) {
    boolean consumedAka = hasRed(tile) && (usesAka || plain(tile) < 2);
    if (consumedAka) {
      removeRed(tile);
    } else {
      shape.remove(tile);
    }
    shape.remove(tile);
    addMeld(Meld.pon(tile, relative(source), akaSource(calledAka, consumedAka)));
  }

  private void daiminkan(int tile, int source, boolean calledAka) {
    boolean consumedAka = removeCopies(tile, Meld.Type.DAIMINKAN.tiles() - 1);
    addMeld(Meld.daiminkan(tile, relative(source), akaSource(calledAka, consumedAka)));
  }

  private void ankan(int tile) {
    addMeld(Meld.ankan(tile, removeCopies(tile, Tiles.COPIES)));
  }

  private void kakan(int tile) {
    boolean addedAka = hasRed(tile);
    if (addedAka) {
      removeRed(tile);
    } else {
      shape.remove(tile);
    }
    for (int index = 0; index < meldCount; index++) {
      Meld pon = melds[index];
      if (pon.type() == Meld.Type.PON && pon.base() == tile) {
        melds[index] = Meld.kakan(pon, addedAka);
        meldDora += tiles.dora(tile);
        if (addedAka) {
          ownedRed |= Tiles.bit(tile);
        }
        return;
      }
    }
    throw new IllegalStateException("kakan without its pon: tile " + tile);
  }

  /** 鳴いた牌を出した席から、面子の鳴き元（自分から見た相対の席）を求める。 */
  private Meld.RelativeSource relative(int source) {
    return Meld.RelativeSource.of(0, source, seats);
  }

  /** 赤の出どころ。鳴いた牌が赤なら鳴いた牌、手の赤を使ったなら手の牌、どちらでもなければ赤なし。 */
  private static Meld.AkaSource akaSource(boolean calledAka, boolean consumedAka) {
    if (calledAka) {
      return Meld.AkaSource.CALLED_TILE;
    }
    return consumedAka ? Meld.AkaSource.CONSUMED_HAND_TILE : Meld.AkaSource.NONE;
  }

  /** 手の赤を先に使って、その牌を {@code copies} 枚除く。赤を使ったなら true を返す。 */
  private boolean removeCopies(int tile, int copies) {
    boolean aka = hasRed(tile);
    if (aka) {
      removeRed(tile);
      copies--;
    }
    for (int copy = 0; copy < copies; copy++) {
      shape.remove(tile);
    }
    return aka;
  }

  private boolean hasRed(int tile) {
    return Tiles.contains(concealedRed, tile);
  }

  /** 純手牌にある、その種類の赤でない牌の枚数。 */
  private int plain(int tile) {
    return shape.count(tile) - (hasRed(tile) ? 1 : 0);
  }

  private void removeRed(int tile) {
    shape.remove(tile);
    long bit = Tiles.bit(tile);
    concealedRed &= ~bit;
    ownedRed &= ~bit;
  }

  private void addMeld(Meld meld) {
    melds[meldCount++] = meld;
    shape.addMeld();
    menzen &= meld.type().concealed();
    for (int tile = 0, size = meld.size(); tile < size; tile++) {
      meldDora += tiles.dora(meld.tile(tile));
    }
    if (meld.red()) {
      ownedRed |= meld.kinds() & Tiles.FIVES;
    }
  }
}
