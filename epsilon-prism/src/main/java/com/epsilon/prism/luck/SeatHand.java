package com.epsilon.prism.luck;

import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.RoundLimits;
import com.epsilon.core.Tile;
import com.epsilon.prism.Limits;
import java.util.Objects;

/**
 * 観測 O° の 1 席の手（34 の枚数・赤・副露）。形の解析と和了の採点にそのまま渡せるように {@link HandView} を実装する。付け替えて使い回し、
 * 割り当てをしない。
 */
final class SeatHand implements HandView {

  private static final int MELD_ID_BITS =
      Integer.SIZE - Integer.numberOfLeadingZeros(Limits.MELD_IDS - 1);

  private final HandShapeState shape = new HandShapeState();
  private final int[] counts = new int[Tile.NUM_TILE_TYPES];
  private final int[] owned = new int[Tile.NUM_TILE_TYPES];
  private final Meld[] melds = new Meld[RoundLimits.MAX_MELDS];
  private int concealedAka;
  private int ownedAka;
  private int kans;
  private long signature;
  private boolean menzen;

  /**
   * O° の相対席 {@code seat} の手にする。
   *
   * @param o 観測 O°（付け替えた読み手）
   * @param seat 相対席
   */
  void load(LuckObservation o, int seat) {
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      counts[tile] = o.handCount(seat, tile);
      owned[tile] = counts[tile];
    }
    int meldCount = o.meldCount(seat);
    shape.load(counts, meldCount);
    concealedAka = o.handAkaMask(seat);
    ownedAka = concealedAka;
    kans = 0;
    signature = 0L;
    menzen = true;
    for (int index = 0; index < meldCount; index++) {
      Meld meld = o.meld(seat, index);
      melds[index] = meld;
      signature |= (long) meld.canonicalId() << (index * MELD_ID_BITS);
      menzen &= meld.preservesMenzen();
      kans += meld.isKan() ? 1 : 0;
      for (int at = 0, size = meld.size(); at < size; at++) {
        owned[meld.tileAt(at)]++;
      }
      if (meld.containsAkaTile()) {
        ownedAka = AkaTileMask.includeTile(ownedAka, meld.akaTileType());
      }
    }
  }

  /** 手の枚数だけの形（副露の数を含む）。解析と採点にはこの具体の型を渡す。 */
  HandShapeState shape() {
    return shape;
  }

  /** 手と副露の牌 {@code tile} の枚数。 */
  int owned(int tile) {
    return owned[tile];
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
