package com.epsilon.mahjong.client;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/** テスト用の偽の接続。用意したメッセージを順に読ませ、送られたメッセージを記録する。読ませるメッセージが尽きると接続が切れる。 */
public final class FakeTransport implements Transport {

  private final ArrayDeque<String> frames;
  private final List<String> sent = new ArrayList<>();
  private boolean closed;

  public FakeTransport(List<String> frames) {
    this.frames = new ArrayDeque<>(frames);
  }

  @Override
  public Frame read() throws IOException {
    if (closed || frames.isEmpty()) {
      throw new IOException("The fake connection is closed");
    }
    return new Frame(frames.removeFirst(), System.nanoTime());
  }

  @Override
  public synchronized void send(String text) {
    sent.add(text);
  }

  @Override
  public void close() {
    closed = true;
  }

  /** 送られたメッセージを、送られた順に返す。 */
  public synchronized List<String> sent() {
    return List.copyOf(sent);
  }

  public boolean closed() {
    return closed;
  }
}
