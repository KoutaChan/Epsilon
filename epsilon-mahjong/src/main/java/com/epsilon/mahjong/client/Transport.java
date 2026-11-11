package com.epsilon.mahjong.client;

import java.io.IOException;

/** サーバへの 1 本の接続。テキストのメッセージを順に読み書きする。 */
public interface Transport extends AutoCloseable {

  /** 受け取ったメッセージと、受け取り始めた時刻（System.nanoTime の値）。 */
  record Frame(String text, long receivedNanos) {}

  /** 次のメッセージを待って読む。接続が切れていれば IOException を投げる。 */
  Frame read() throws IOException, InterruptedException;

  /** メッセージを 1 つ送る。複数のスレッドから呼んでもメッセージは混ざらない。 */
  void send(String text) throws IOException;

  @Override
  void close();
}
