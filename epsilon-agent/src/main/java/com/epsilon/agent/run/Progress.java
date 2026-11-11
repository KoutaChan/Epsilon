package com.epsilon.agent.run;

/** run の進み具合。state.json の record が実装し、次の反復の番号を {@code next} のフィールドに持つ。 */
public interface Progress {

  /** 次に回す反復の番号（1 から）。反復 {@code next − 1} までを閉じ終え、バージョン {@code next} まで公開している。 */
  int next();
}
