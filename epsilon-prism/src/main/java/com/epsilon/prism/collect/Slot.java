package com.epsilon.prism.collect;

/** 記録の 1 枠: 手 action を打った経路の局末の値を target に渡す。 */
record Slot(Pending target, int action) {}
