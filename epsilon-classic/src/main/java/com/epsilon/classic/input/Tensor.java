package com.epsilon.classic.input;

import com.epsilon.classic.Architecture;
import com.epsilon.classic.ArchitectureScoped;
import java.util.Set;

/**
 * ネットワークへ渡す入力のテンソル。アーキテクチャが使う定数を宣言の順に並べたものが、ネットワークへ渡す順になる。数は Nano・Major
 * とも 12 個。1 行の要素の数は、{@link ClassicSchema} がフィールドの宣言から求める。
 */
public enum Tensor implements ArchitectureScoped {
  /** 局・席・牌・河・面子の整数のフィールド。Major では、その後ろに公開の履歴の要素が続く。 */
  STATE_CATEGORIES(Slab.CATEGORICAL, Axis.ROW),
  /** 合法手の整数のフィールド。 */
  ACTION_CATEGORIES(Slab.CATEGORICAL, Axis.ACTION),
  /** 合法手どうしの位置の対応（同じ牌の打牌と立直の打牌）。 */
  ACTION_ROUTES(Slab.CATEGORICAL, Axis.ACTION),
  /** 絶対席の順に並べた持ち点（100 点単位）。 */
  POINT_LEDGER_100(Slab.CATEGORICAL, Axis.ROW, Architecture.MAJOR),
  /** その合法手で和了したときの点数の情報。 */
  ACTION_WIN_FACTS(Slab.CATEGORICAL, Axis.ACTION, Architecture.MAJOR),
  /** 遷移の整数のフィールド。 */
  TRANSITION_CATEGORIES(Slab.CATEGORICAL, Axis.TRANSITION),
  /** 遷移の後の手の、牌の種類ごとのコード。 */
  TRANSITION_TILES(Slab.CATEGORICAL, Axis.TRANSITION),
  /** 遷移の後の待ち牌の種類 ＋ 1。 */
  WAIT_TILE_IDS(Slab.CATEGORICAL, Axis.TRANSITION),
  /** 遷移の後の待ちごとに、ロン・ツモで付く役の bit をまとめた値。 */
  WAIT_YAKUS(Slab.CATEGORICAL, Axis.TRANSITION, Architecture.NANO),
  /** 遷移の後の待ちごとに、ロン・ツモで和了したときの点数の情報。 */
  WAIT_WIN_FACTS(Slab.CATEGORICAL, Axis.TRANSITION, Architecture.MAJOR),
  /** 局・席・牌・河・面子の実数のフィールド。 */
  STATE_NUMERICS(Slab.NUMERIC, Axis.ROW),
  /** 価値のヘッドだけが読む値。GRP の事前分布と、局の始まりの公開の値。 */
  BOUNDARY_CONTEXT(Slab.NUMERIC, Axis.ROW),
  /** 合法手の実数のフィールド。 */
  ACTION_NUMERICS(Slab.NUMERIC, Axis.ACTION, Architecture.NANO),
  /** 遷移の実数のフィールド。 */
  TRANSITION_NUMERICS(Slab.NUMERIC, Axis.TRANSITION),
  /** 遷移の後の待ちごとに、点数を表す実数のフィールド。 */
  WAIT_SCORES(Slab.NUMERIC, Axis.TRANSITION, Architecture.NANO);

  /** 要素の型。整数は short の連続した配列に、実数は float の連続した配列に置く。 */
  public enum Slab {
    CATEGORICAL,
    NUMERIC
  }

  /** 1 行の中で要素を繰り返す単位。 */
  public enum Axis {
    /** 行に 1 つ。 */
    ROW,
    /** 合法手の軸の大きさだけ繰り返す。 */
    ACTION,
    /** 合法手の軸の大きさ × 遷移の軸の大きさだけ繰り返す。 */
    TRANSITION
  }

  private final Slab slab;
  private final Axis axis;
  private final Set<Architecture> architectures;

  Tensor(Slab slab, Axis axis, Architecture... only) {
    this.slab = slab;
    this.axis = axis;
    architectures = ArchitectureScoped.scope(only);
  }

  public Slab slab() {
    return slab;
  }

  public Axis axis() {
    return axis;
  }

  @Override
  public Set<Architecture> architectures() {
    return architectures;
  }
}
