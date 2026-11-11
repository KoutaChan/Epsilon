package com.epsilon.reviewer.engine.review;

/** 解析が取り消されたかを答え、進み具合を受け取る。 */
public interface JobControl {
  /** 取り消されたか。イベントと判断のたびに確かめる。 */
  boolean cancelled();

  /** 進み具合を百分率（0〜99）で知らせる。手順を書くたびに呼ぶ。 */
  void progress(int percent);
}
