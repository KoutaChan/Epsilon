package com.epsilon.agent.eval;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;

/** 王者との決闘で「悪い」が続いたときの扱い。 */
@Choice(key = "kind", doc = "「悪い」が続いたときにすること")
public sealed interface OnWorseSettings {

  /** 「悪い」が何回続いたら扱いを行うか。 */
  int streak();

  /** アラートを記録に書いて、そのまま続ける。 */
  @Case("REPORT")
  record Report(@Key(doc = "アラートを書くまでに「悪い」が続く回数", value = "2") @Range(min = 1) int streak)
      implements OnWorseSettings {}

  /** 決闘のジョブが run に一時停止の要求を出す。 */
  @Case("PAUSE")
  record Pause(@Key(doc = "一時停止を要求するまでに「悪い」が続く回数", presence = Presence.REQUIRED) @Range(min = 1) int streak)
      implements OnWorseSettings {}

  /**
   * 次の反復を、王者のバージョンのネットワークから始める。直近 {@code span} バージョンの中で戻した回数が {@code maxRollbacks} に
   * 達していれば、戻さずに一時停止する。
   *
   * @param streak 戻すまでに「悪い」が続く回数
   * @param maxRollbacks {@code span} の中で戻してよい回数
   * @param span 戻した回数を数えるバージョンの幅
   */
  @Case("ROLLBACK")
  record Rollback(
      @Key(doc = "戻すまでに「悪い」が続く回数", presence = Presence.REQUIRED) @Range(min = 1) int streak,
      @Key(doc = "span の中で戻してよい回数", presence = Presence.REQUIRED) @Range(min = 1) int maxRollbacks,
      @Key(doc = "戻した回数を数えるバージョンの幅", presence = Presence.REQUIRED) @Range(min = 1) int span)
      implements OnWorseSettings {}
}
