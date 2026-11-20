package com.epsilon.prism.input;

import com.epsilon.mahjong.rules.Variant;

/** 入力のイベント容量とスキーマ記述の名前。 */
public enum InputRules {
  FOUR_PLAYER(128, "epsilon-prism-v2-input"),
  THREE_PLAYER(96, "epsilon-prism-3p-input");

  final int eventCapacity;
  final String schemaName;

  InputRules(int eventCapacity, String schemaName) {
    this.eventCapacity = eventCapacity;
    this.schemaName = schemaName;
  }

  public static InputRules of(Variant variant) {
    return switch (variant) {
      case TENHOU_4P -> FOUR_PLAYER;
      case TENHOU_3P -> THREE_PLAYER;
    };
  }
}
