package com.epsilon.prism.collect;

import com.epsilon.agent.play.TableStage;
import com.epsilon.prism.luck.LedgerTracker;

/**
 * 収集の時間を測る段。名前は反復の記録の collection.stages のキーなので変えない。卓のコアが測る段（{@link TableStage}）も、
 * ここの名前に直して記録に書く（{@link #of}）。
 */
public enum CollectStage {
  /** 主経路の進行（進行ワーカーで測る）。 */
  MAIN_ADVANCE,
  /** 写しの進行。 */
  COPY_ADVANCE,
  /** 分岐の進行。 */
  FORK_ADVANCE,
  /** 候補集合と逸脱の計算（進行ワーカーで測る）。 */
  CANDIDATES,
  /** その場で求める方策の評価。 */
  INLINE_POLICY,
  DRIVER_BUSY,
  DRIVER_WAIT,
  /**
   * 運の精算の帳簿の計算（{@link LedgerTracker}）。主経路・写し・分岐の進行の時間の内数で、回数は呼び出しの数。帳簿を付けない収集では
   * 数えないので、記録に出ない。
   */
  LEDGER,
  /** 記録 O を取り込む時間（進行の時間の内数）。回数は取り込んだ O の数。 */
  V2_CAPTURE,
  /**
   * 行を作る判断の記録（O の写しと Oracle）。ここから下の段は行を書く収集でだけ数える。進行の中で測る段は、進行の時間の内数になる。
   */
  V2_ORACLE,
  /** 聴牌への到達の追跡。値を待つ判断を持つ席の打牌ごとに数える。 */
  V2_REACH,
  /** 局末の事実。値を待つ判断がある経路の局末ごとに数える。 */
  V2_HAND_END,
  /** 局末の 3 ワード（ドライバで測る）。回数は値を待つ判断の数。 */
  V2_OUTCOMES,
  /** 行を作ってライターに渡す（ドライバで、行ごとに測る）。圧縮はライターのスレッドで行うので、ここに入らない。 */
  V2_ROWS;

  /** 卓のコアの段を、記録での名前に直す。 */
  public static CollectStage of(TableStage stage) {
    return switch (stage) {
      case INLINE_POLICY -> INLINE_POLICY;
      case CAPTURE -> V2_CAPTURE;
      case DRIVER_BUSY -> DRIVER_BUSY;
      case DRIVER_WAIT -> DRIVER_WAIT;
    };
  }
}
