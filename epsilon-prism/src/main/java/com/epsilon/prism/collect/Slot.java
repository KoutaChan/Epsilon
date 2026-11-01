package com.epsilon.prism.collect;

/**
 * 記録の 1 枠: 手 action を打った経路の局末の値を target に渡す。ledger は運の精算の帳簿（{@link EventLedger}。長さ {@value
 * EventLedger#FEATURES}、帳簿を付ける収集だけ。ほかは null）。帳簿はその枠の経路の進行だけが書き、局末の値と一緒にドライバへ渡す。
 */
record Slot(Pending target, int action, double[] ledger) {}
