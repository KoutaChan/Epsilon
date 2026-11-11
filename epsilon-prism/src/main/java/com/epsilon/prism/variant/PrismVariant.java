package com.epsilon.prism.variant;

import com.epsilon.agent.app.Command;
import com.epsilon.agent.settings.Layer;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.eval.OpponentRules;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.learn.TargetWriter;
import com.epsilon.prism.model.AuxiliaryLoss;
import java.util.Collection;
import java.util.List;

/**
 * ルール（四人打ち・三人打ち）ごとの PRISM のモジュールが実装するインターフェース。卓のコア・ネットワーク・学習は、ルールごとの形式を
 * このインターフェースから受け取り、ルールごとのモジュールの型を直接は読まない。実装は ServiceLoader で登録し、ルールで引く。
 */
public interface PrismVariant {

  /**
   * ルールのモジュールを返す。
   *
   * @param variant ルール
   * @return モジュール
   * @throws IllegalStateException そのルールのモジュールが classpath に無いとき
   */
  static PrismVariant of(Variant variant) {
    PrismVariant found = PrismVariants.LOADED.get(variant);
    if (found == null) {
      throw new IllegalStateException("No PRISM module for " + variant + " on the classpath");
    }
    return found;
  }

  /**
   * ルールのモジュールが classpath にあるかを返す。
   *
   * @param variant ルール
   * @return あれば true
   */
  static boolean present(Variant variant) {
    return PrismVariants.LOADED.containsKey(variant);
  }

  /** classpath にある全てのルールのモジュールを、ルールの順に返す。 */
  static Collection<PrismVariant> all() {
    return PrismVariants.LOADED.values();
  }

  /** このモジュールのルール。 */
  Variant variant();

  /** ネットワークの入力の形式。 */
  InputFormat input();

  /**
   * 補助のヘッドの教師を書くライターを作る。エンコードするスレッドごとに 1 つ作る。
   *
   * @param loss 補助のヘッドの損失（重みが正のグループにだけ教師を作る）
   * @return ライター
   */
  TargetWriter targets(AuxiliaryLoss loss);

  /** 評価の卓で相手として受け付けるモデル。 */
  OpponentRules opponents();

  /**
   * このルールだけで使うコマンドを表す sealed のインターフェースを、使い方を表示する順に返す。これらのコマンドは、アプリ全体の設定を
   * 読まない。
   */
  List<Class<? extends Command<Record>>> commands();

  /**
   * ルールごとの既定値の層。四人打ちと三人打ちで既定値が違うキー（局の段階のしきい値、北抜きのクラス、クラスをキーにした率の表）だけを書く。
   * 中身は resources の {@code prism/variants/<ルールの名前>.toml}。
   */
  default Layer settings() {
    return Layer.resource("prism/variants/" + variant().name() + ".toml", "variant " + variant().name());
  }
}
