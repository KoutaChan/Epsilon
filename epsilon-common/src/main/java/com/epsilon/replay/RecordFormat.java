package com.epsilon.replay;

/** 生牌譜の提供元。圧縮と符号化は読込処理が扱う。 */
public enum RecordFormat {
  TENHOU("tenhou"),
  MJAI("mjai"),
  MAHJONG_SOUL("majsoul");
  private final String sourceId;

  RecordFormat(String sourceId) {
    this.sourceId = sourceId;
  }

  public String sourceId() {
    return sourceId;
  }
}
