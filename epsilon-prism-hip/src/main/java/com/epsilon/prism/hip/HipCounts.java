package com.epsilon.prism.hip;

/**
 * 四人打ちか三人打ちかで決まる数。1 行のトークンは局 1 個、席、牌の順に並び、その後にイベントが続く。
 *
 * @param seats 行ごとの席のトークンの数
 * @param tiles 行ごとの牌のトークンの数（牌の種類の数。牌の関係の表はその 2 乗の大きさ）
 * @param keyCodes 遷移のキーのコードの数（候補層でキーとバリューに足す差分の表の行の数）
 * @param tileRelations 牌の関係の種類の数
 * @param maxEvents 行ごとのイベントのトークンの上限
 */
public record HipCounts(int seats, int tiles, int keyCodes, int tileRelations, int maxEvents) {

  /** 関係のコードのタイルの幅（トークンの数）。 */
  public static final int CODE_BLOCK = 16;

  /**
   * 数を確かめる。
   *
   * @throws IllegalArgumentException 席・牌・遷移のキーのコード・牌の関係の種類のどれかの数が 0 以下のとき、またはイベントの上限が負の
   *     とき
   */
  public HipCounts {
    if (seats <= 0 || tiles <= 0 || keyCodes <= 0 || tileRelations <= 0 || maxEvents < 0) {
      throw new IllegalArgumentException(
          "invalid counts: seats=" + seats + " tiles=" + tiles + " keyCodes=" + keyCodes + " tileRelations="
              + tileRelations + " maxEvents=" + maxEvents);
    }
  }

  /**
   * 行ごとに必ずあるトークン（局・席・牌）の数を返す。
   *
   * @return 1 + 席 + 牌
   */
  public int fixed() {
    return 1 + seats + tiles;
  }

  /**
   * 相手の席の数を返す。
   *
   * @return 席 − 1
   */
  public int opponents() {
    return seats - 1;
  }

  /**
   * 候補層の 2 段目が引くコードの表の行数を返す。行は、遷移のコード、牌の関係を持たない行 1 つ、牌の関係の種類の順に並ぶ。
   *
   * @return 遷移のコードの数 + 1 + 牌の関係の種類の数
   */
  public int rereadCodes() {
    return keyCodes + 1 + tileRelations;
  }

  /**
   * 1 行のトークンの上限を返す。
   *
   * @return 局・席・牌のトークンの数 + イベントの上限
   */
  public int rowTokens() {
    return fixed() + maxEvents;
  }

  /**
   * 関係のコードを 1 行あたりいくつのタイルに分けるかを返す。
   *
   * @return ⌈行のトークンの上限 / 16⌉
   */
  public int codeBlocks() {
    return Math.ceilDiv(rowTokens(), CODE_BLOCK);
  }
}
