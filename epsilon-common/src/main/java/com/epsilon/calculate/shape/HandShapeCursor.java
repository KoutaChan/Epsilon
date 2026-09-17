package com.epsilon.calculate.shape;

/**
 * 探索中の仮の手牌を、牌種ごとの枚数として保持する再利用可能な作業領域。物理牌IDは保持しない。
 *
 * <p>{@code load}で読み込んだ後は元の手牌から独立する。入れ子の探索では、追加した牌を{@code
 * finally}で必ず取り除き、探索前の状態に戻す。同じインスタンスを複数スレッドから同時に使用してはならない。
 */
public final class HandShapeCursor implements HandShapeView {
  private final HandShapeState state = new HandShapeState();

  @Override
  public void copyShapeInto(HandShapeState destination) {
    state.copyShapeInto(destination);
  }

  @Override
  public void copyTileCountsInto(HandShapeState.TileCountBuffer destination) {
    state.copyTileCountsInto(destination);
  }

  public void load(HandShapeView source) {
    state.load(source);
  }

  public void load(int[] counts, int meldCount) {
    state.load(counts, meldCount);
  }

  public void setMeldCount(int meldCount) {
    state.setMeldCount(meldCount);
  }

  public void add(int tileType) {
    state.add(tileType);
  }

  public void remove(int tileType) {
    state.remove(tileType);
  }

  @Override
  public int count(int tileType) {
    return state.count(tileType);
  }

  @Override
  public int concealedTileCount() {
    return state.concealedTileCount();
  }

  @Override
  public long concealedTileTypeMask() {
    return state.concealedTileTypeMask();
  }

  @Override
  public long concealedPairTileTypeMask() {
    return state.concealedPairTileTypeMask();
  }

  @Override
  public int distinctConcealedTileTypeCount() {
    return state.distinctConcealedTileTypeCount();
  }

  @Override
  public int concealedPairTileTypeCount() {
    return state.concealedPairTileTypeCount();
  }

  @Override
  public int kokushiTileTypeCount() {
    return state.kokushiTileTypeCount();
  }

  @Override
  public int kokushiPairTileTypeCount() {
    return state.kokushiPairTileTypeCount();
  }

  @Override
  public int meldCount() {
    return state.meldCount();
  }
}
