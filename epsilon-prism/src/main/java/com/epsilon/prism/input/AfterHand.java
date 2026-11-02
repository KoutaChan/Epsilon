package com.epsilon.prism.input;

import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.core.Action;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.RoundLimits;
import com.epsilon.core.Tile;
import com.epsilon.prism.Limits;
import com.epsilon.prism.observe.Observation;
import java.util.Objects;

/**
 * 手を打った後の自分の手（34 の枚数・赤・副露）。判断の時点の手（{@link #loadCurrent}）を写し、手と鳴いた後の打牌を当てて作る。形の解析と
 * 待ちの採点にそのまま渡せるように {@link HandView} を実装する。割り当てをしない。
 *
 * <p>手の当て方はエンジンの射影（{@code ActionEffect}）と同じ決まりにする: 打牌は赤を選ぶか赤しかないときに赤を捨てる、チー・ポンは手の赤を
 * 選ぶか通常の牌が足りないときに赤を使う、大明槓・暗槓は手の赤を先に使う、加槓は手の赤があれば赤を足す。副露の元と赤の由来は直前の打牌
 * （O の直前の出来事）から決める。和了・九種九牌・見送りは手を変えない。副露は成立の順に持ち、加槓はポンを置き換える。
 *
 * <p>副露の牌の表ドラは、判断の時点の手で 1 回数え、副露を足すときにその副露の分だけ足す（遷移ごとに数え直さない）。
 */
final class AfterHand implements HandView {

  private static final int MELD_ID_BITS =
      Integer.SIZE - Integer.numberOfLeadingZeros(Limits.MELD_IDS - 1);

  private final HandShapeState shape = new HandShapeState();
  private final int[] counts = new int[Tile.NUM_TILE_TYPES];
  private final Meld[] melds = new Meld[RoundLimits.MAX_MELDS];
  private TileCounts tiles;
  private int concealedAka;
  private int ownedAka;
  private int kans;
  private int meldDora;
  private long signature;
  private boolean menzen;

  /**
   * 判断の時点の自分の手にする。
   *
   * @param o 判断の記録（手の赤を読む）
   * @param tiles 牌ごとの枚数（手の枚数とドラの倍率を読む。この判断の間、この手と写した手が使う）
   * @param events O の出来事（自分の今の副露を読む）
   */
  void loadCurrent(Observation o, TileCounts tiles, EventTable events) {
    this.tiles = tiles;
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      counts[tile] = tiles.hand(tile);
    }
    int meldCount = events.meldCount(0);
    shape.load(counts, 0);
    concealedAka = o.handAkaMask();
    ownedAka = concealedAka;
    kans = 0;
    meldDora = 0;
    signature = 0L;
    menzen = true;
    for (int index = 0; index < meldCount; index++) {
      addMeld(events.meld(0, index));
    }
  }

  /**
   * 別の手を写す。
   *
   * @param source 写す元
   */
  void copyFrom(AfterHand source) {
    source.shape.copyShapeInto(shape);
    System.arraycopy(source.melds, 0, melds, 0, RoundLimits.MAX_MELDS);
    tiles = source.tiles;
    concealedAka = source.concealedAka;
    ownedAka = source.ownedAka;
    kans = source.kans;
    meldDora = source.meldDora;
    signature = source.signature;
    menzen = source.menzen;
  }

  /**
   * 合法手を当てる。チー・ポンの後の打牌は {@link #discard} で別に当てる。
   *
   * @param action 合法手
   * @param o 判断の記録（鳴く打牌の席・牌・赤を直前の出来事から読む）
   */
  void apply(Action action, Observation o) {
    int tile = action.tileType();
    switch (action.type()) {
      case DAHAI, RIICHI_DAHAI -> discard(action);
      case CHI -> chi(action, o.lastEventTile(), o.lastEventAka());
      case PON -> pon(tile, action.usesAkaTileFromHand(), o.lastEventSeat(), o.lastEventAka());
      case DAIMINKAN -> daiminkan(tile, o.lastEventSeat(), o.lastEventAka());
      case ANKAN -> ankan(tile);
      case KAKAN -> kakan(tile);
      case TSUMO_AGARI, RON_AGARI, KYUSHU_KYUHAI, PASS -> {}
    }
  }

  /**
   * 打牌を当てる（赤を選ぶか、赤しかないときに赤を捨てる）。
   *
   * @param discard 打牌の手
   */
  void discard(Action discard) {
    int tile = discard.tileType();
    if (hasAkaTile(tile) && (discard.usesAkaTileFromHand() || !hasNonAkaTile(tile))) {
      removeAka(tile);
    } else {
      shape.remove(tile);
    }
  }

  /**
   * 手の枚数だけの形（副露の数を含む）を返す。形の解析と採点には、この具体の型を渡す（手の型ごとに分かれた呼び出しを避ける）。
   *
   * @return 形（読むだけ）
   */
  HandShapeState shape() {
    return shape;
  }

  /**
   * 手と副露のドラ（表ドラと赤、裏なし）を返す。
   *
   * @return ドラの数
   */
  int dora() {
    int dora = Integer.bitCount(ownedAka) + meldDora;
    for (long rest = shape.concealedTileTypeMask() & tiles.doraTiles(); rest != 0L; rest &= rest - 1) {
      int tile = Long.numberOfTrailingZeros(rest);
      dora += shape.count(tile) * tiles.dora(tile);
    }
    return dora;
  }

  private void chi(Action action, int calledTile, boolean calledAka) {
    int base = action.chiSequenceBaseTileType();
    boolean consumedAka = false;
    for (int tile = base; tile <= base + 2; tile++) {
      if (tile == calledTile) {
        continue;
      }
      if (hasAkaTile(tile) && (action.usesAkaTileFromHand() || !hasNonAkaTile(tile))) {
        removeAka(tile);
        consumedAka = true;
      } else {
        shape.remove(tile);
      }
    }
    addMeld(Meld.chi(base, calledTile, Meld.AkaSource.forCall(calledAka, consumedAka)));
  }

  private void pon(int tile, boolean usesAka, int source, boolean calledAka) {
    boolean consumedAka = hasAkaTile(tile) && (usesAka || !hasMultipleNonAkaTiles(tile));
    if (consumedAka) {
      removeAka(tile);
    } else {
      shape.remove(tile);
    }
    shape.remove(tile);
    addMeld(
        Meld.pon(
            tile,
            Meld.RelativeSource.fromPlayerOffset(source),
            Meld.AkaSource.forCall(calledAka, consumedAka)));
  }

  private void daiminkan(int tile, int source, boolean calledAka) {
    boolean consumedAka = removeCopies(tile, 3);
    addMeld(
        Meld.daiminkan(
            tile,
            Meld.RelativeSource.fromPlayerOffset(source),
            Meld.AkaSource.forCall(calledAka, consumedAka)));
  }

  private void ankan(int tile) {
    addMeld(Meld.ankan(tile, removeCopies(tile, Tile.TILES_PER_TYPE)));
  }

  private void kakan(int tile) {
    boolean addedAka = hasAkaTile(tile);
    if (addedAka) {
      removeAka(tile);
    } else {
      shape.remove(tile);
    }
    for (int index = 0, meldCount = meldCount(); index < meldCount; index++) {
      Meld pon = melds[index];
      if (pon.type() == Meld.Type.PON && pon.baseTileType() == tile) {
        Meld kakan = Meld.kakan(pon, addedAka);
        melds[index] = kakan;
        long mask = ((1L << MELD_ID_BITS) - 1) << (index * MELD_ID_BITS);
        signature = signature & ~mask | (long) kakan.canonicalId() << (index * MELD_ID_BITS);
        kans++;
        meldDora += tiles.dora(tile);
        if (addedAka) {
          ownedAka = AkaTileMask.includeTile(ownedAka, tile);
        }
        return;
      }
    }
    throw new IllegalStateException("kakan without its pon: tile " + tile);
  }

  /** 手の赤を先に使って copies 枚を除き、赤を使ったなら true を返す。 */
  private boolean removeCopies(int tile, int copies) {
    boolean aka = hasAkaTile(tile);
    if (aka) {
      removeAka(tile);
      copies--;
    }
    for (int copy = 0; copy < copies; copy++) {
      shape.remove(tile);
    }
    return aka;
  }

  private void removeAka(int tile) {
    shape.remove(tile);
    concealedAka = AkaTileMask.excludeTile(concealedAka, tile);
    ownedAka = AkaTileMask.excludeTile(ownedAka, tile);
  }

  private void addMeld(Meld meld) {
    int index = meldCount();
    melds[index] = meld;
    signature |= (long) meld.canonicalId() << (index * MELD_ID_BITS);
    shape.setMeldCount(index + 1);
    menzen &= meld.preservesMenzen();
    kans += meld.isKan() ? 1 : 0;
    for (int tile = 0, size = meld.size(); tile < size; tile++) {
      meldDora += tiles.dora(meld.tileAt(tile));
    }
    if (meld.containsAkaTile()) {
      ownedAka = AkaTileMask.includeTile(ownedAka, meld.akaTileType());
    }
  }

  @Override
  public void copyShapeInto(HandShapeState destination) {
    shape.copyShapeInto(destination);
  }

  @Override
  public void copyTileCountsInto(HandShapeState.TileCountBuffer destination) {
    shape.copyTileCountsInto(destination);
  }

  @Override
  public int count(int tileType) {
    return shape.count(tileType);
  }

  @Override
  public int concealedTileCount() {
    return shape.concealedTileCount();
  }

  @Override
  public int meldCount() {
    return shape.meldCount();
  }

  @Override
  public long concealedTileTypeMask() {
    return shape.concealedTileTypeMask();
  }

  @Override
  public long concealedPairTileTypeMask() {
    return shape.concealedPairTileTypeMask();
  }

  @Override
  public boolean hasAkaTile(int tileType) {
    return AkaTileMask.containsTile(concealedAka, tileType);
  }

  @Override
  public int concealedAkaMask() {
    return concealedAka;
  }

  @Override
  public int ownedAkaMask() {
    return ownedAka;
  }

  @Override
  public Meld meld(int meldIndex) {
    return melds[Objects.checkIndex(meldIndex, meldCount())];
  }

  @Override
  public int meldCallAfterRiverIndex(int meldIndex) {
    return -1;
  }

  @Override
  public int meldKanAfterRiverIndex(int meldIndex) {
    return -1;
  }

  @Override
  public boolean isMenzen() {
    return menzen;
  }

  @Override
  public int kanCount() {
    return kans;
  }

  @Override
  public long canonicalMeldSignature() {
    return signature;
  }
}
