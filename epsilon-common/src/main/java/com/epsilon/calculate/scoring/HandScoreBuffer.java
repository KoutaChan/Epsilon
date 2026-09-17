package com.epsilon.calculate.scoring;

/** 呼び出し元が所有する再利用結果。次の評価まで有効。 保存する場合は {@link #snapshot()} で不変の値へ変換する。 */
public final class HandScoreBuffer {
  private boolean menzen;
  private long yakuBits;
  private int han;
  private int fu;
  private int omoteDoraCount;
  private int akaDoraCount;
  private int uraDoraCount;
  private int yakumanMultiplier;
  private int basePoints;

  public HandScoreBuffer() {}

  public static HandScoreBuffer of(
      boolean menzen, long yakuBits, int omoteDora, int akaDora, int uraDora, int fu) {
    return new HandScoreBuffer().set(menzen, yakuBits, omoteDora, akaDora, uraDora, fu);
  }

  public HandScoreBuffer set(
      boolean menzen, long bits, int omoteDora, int akaDora, int uraDora, int fu) {
    if (omoteDora < 0 || akaDora < 0 || uraDora < 0 || fu < 0)
      throw new IllegalArgumentException("negative score component");
    clear();
    this.menzen = menzen;
    yakuBits = YakuBits.effective(bits, menzen);
    yakumanMultiplier = YakuBits.yakumanCount(yakuBits);
    if (yakumanMultiplier > 0) {
      basePoints = 8000 * yakumanMultiplier;
    } else if (yakuBits != 0L) {
      omoteDoraCount = omoteDora;
      akaDoraCount = akaDora;
      uraDoraCount = uraDora;
      han = YakuBits.han(yakuBits, menzen) + omoteDora + akaDora + uraDora;
      this.fu = fu;
      basePoints = ScoreMath.normalBasePoints(han, fu);
    }
    return this;
  }

  /** 評価器が比較済みの候補を保存する。翻・基本点を再計算しない。 */
  void accept(
      boolean menzen,
      long bits,
      int han,
      int fu,
      int omote,
      int aka,
      int ura,
      int multiplier,
      int points) {
    this.menzen = menzen;
    yakuBits = bits;
    this.han = han;
    this.fu = fu;
    omoteDoraCount = omote;
    akaDoraCount = aka;
    uraDoraCount = ura;
    yakumanMultiplier = multiplier;
    basePoints = points;
  }

  public void clear() {
    menzen = false;
    yakuBits = 0L;
    han = fu = omoteDoraCount = akaDoraCount = uraDoraCount = yakumanMultiplier = basePoints = 0;
  }

  public HandScoreBuffer copyFrom(HandScoreBuffer source) {
    menzen = source.menzen;
    yakuBits = source.yakuBits;
    han = source.han;
    fu = source.fu;
    omoteDoraCount = source.omoteDoraCount;
    akaDoraCount = source.akaDoraCount;
    uraDoraCount = source.uraDoraCount;
    yakumanMultiplier = source.yakumanMultiplier;
    basePoints = source.basePoints;
    return this;
  }

  public HandScoreBuffer copyFrom(HandScore source) {
    menzen = source.isMenzen();
    yakuBits = source.yakuBits();
    han = source.han();
    fu = source.fu();
    omoteDoraCount = source.omoteDoraCount();
    akaDoraCount = source.akaDoraCount();
    uraDoraCount = source.uraDoraCount();
    yakumanMultiplier = source.yakumanMultiplier();
    basePoints = source.basePoints();
    return this;
  }

  public HandScore snapshot() {
    return new HandScore(
        menzen,
        yakuBits,
        han,
        fu,
        omoteDoraCount,
        akaDoraCount,
        uraDoraCount,
        yakumanMultiplier,
        basePoints);
  }

  public boolean available() {
    return yakuBits != 0L;
  }

  public boolean isMenzen() {
    return menzen;
  }

  public long yakuBits() {
    return yakuBits;
  }

  public int han() {
    return han;
  }

  public int fu() {
    return fu;
  }

  public int omoteDoraCount() {
    return omoteDoraCount;
  }

  public int akaDoraCount() {
    return akaDoraCount;
  }

  public int uraDoraCount() {
    return uraDoraCount;
  }

  public int yakumanMultiplier() {
    return yakumanMultiplier;
  }

  public int basePoints() {
    return basePoints;
  }

  public boolean isYakuman() {
    return yakumanMultiplier > 0;
  }

  public boolean hasYaku(ScoringYaku yaku) {
    return YakuBits.contains(yakuBits, yaku);
  }

  public boolean hasYakuman(ScoringYaku yaku) {
    return yaku.isYakuman() && hasYaku(yaku);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof HandScoreBuffer other)) return false;
    return menzen == other.menzen
        && yakuBits == other.yakuBits
        && han == other.han
        && fu == other.fu
        && omoteDoraCount == other.omoteDoraCount
        && akaDoraCount == other.akaDoraCount
        && uraDoraCount == other.uraDoraCount
        && yakumanMultiplier == other.yakumanMultiplier
        && basePoints == other.basePoints;
  }

  @Override
  public int hashCode() {
    int result = Boolean.hashCode(menzen);
    result = 31 * result + Long.hashCode(yakuBits);
    result = 31 * result + han;
    result = 31 * result + fu;
    result = 31 * result + omoteDoraCount;
    result = 31 * result + akaDoraCount;
    result = 31 * result + uraDoraCount;
    result = 31 * result + yakumanMultiplier;
    return 31 * result + basePoints;
  }

  @Override
  public String toString() {
    return snapshot().toString();
  }
}
