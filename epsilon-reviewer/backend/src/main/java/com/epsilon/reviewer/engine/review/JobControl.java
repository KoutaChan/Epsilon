package com.epsilon.reviewer.engine.review;

/** 解析の取り消しと進み。 */
public interface JobControl {
  /** 取り消されたか。出来事と判断のたびに見る。 */
  boolean cancelled();

  /** 進み（0..99）。出来事を適用するたびに知らせる。 */
  void progress(int percent);
}
