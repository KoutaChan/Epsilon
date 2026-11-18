package com.epsilon.agent.settings;

/** キーを変えてよい時期。どれも、設定の行を足した後に始まる最初の反復か、起動か、ジョブから効く。 */
public enum Change {
  /** run の間は変えない。 */
  FIXED("run の間は変えない"),
  /** 次に始まる反復の収集から効く。集める行が変わる。 */
  COLLECT("次の反復の収集から効く"),
  /** 次に始まる反復の学習から効く。保存したコホートで学習し直すときにも変えられる。 */
  FIT("次の反復の学習から効く"),
  /** 次の起動から効く。結果の分布は変えないが、ビット単位では変わりうる。 */
  MACHINE("次の起動から効く"),
  /** 次に公開するバージョンのジョブから効く。 */
  SCHEDULE("次に公開するバージョンのジョブから効く"),
  /** 親のキーに従う。いちばん外のキーには使えない。 */
  INHERIT(null);

  private final String label;

  Change(String label) {
    this.label = label;
  }

  String label() {
    return label;
  }

  static Change resolve(Key key, Change parent) {
    return key == null || key.change() == INHERIT ? parent : key.change();
  }

  static Change required(Key key, Change parent, String path) {
    Change change = resolve(key, parent);
    if (change == null)
      throw new IllegalStateException("Top-level setting " + path + " must say when it may change");
    return change;
  }
}
