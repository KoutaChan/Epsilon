package com.epsilon.prism.model;

/** ネットワークの形式。model.json の architecture に名前を書く。形の型（{@link PrismShape#architecture()}）が形式を決める。 */
public enum PrismArchitecture {
  /** 型ごとの FFN を持たない形（{@link PrismShape.V2}）。 */
  V2("epsilon-prism-v2"),
  /** 型ごとの FFN を持つ形（{@link PrismShape.V3}）。 */
  V3("epsilon-prism-v3");

  private final String id;

  PrismArchitecture(String id) {
    this.id = id;
  }

  /**
   * model.json に書く名前を返す。
   *
   * @return 名前
   */
  public String id() {
    return id;
  }

  /**
   * 名前から形式を返す。
   *
   * @param id model.json の architecture
   * @return 形式
   * @throws IllegalArgumentException 名前がどの形式でもないとき
   */
  public static PrismArchitecture byId(String id) {
    for (PrismArchitecture architecture : values()) {
      if (architecture.id.equals(id)) {
        return architecture;
      }
    }
    throw new IllegalArgumentException("unknown PRISM architecture " + id);
  }
}
