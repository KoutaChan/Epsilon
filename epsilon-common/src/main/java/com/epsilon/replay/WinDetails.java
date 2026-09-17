package com.epsilon.replay;

import java.util.List;

/**
 * 元牌譜に記録された和了の採点内訳。未記録の数値と配列は null とし、推定値で補わない。
 *
 * @param han 通常役の合計翻数
 * @param fu 符数
 * @param points 本場と供託を含まない和了点
 * @param yakuman 役満倍率。通常の和了は0
 * @param yaku 翻数または役満倍率が付いた役とドラ
 * @param uraIndicators 裏ドラ表示牌の共通牌表記
 */
public record WinDetails(
    Integer han,
    Integer fu,
    Integer points,
    Integer yakuman,
    List<Yaku> yaku,
    List<String> uraIndicators) {

  /** 翻役と役満を区別した役内訳。コードは牌譜形式に依存しない識別子。 */
  public record Yaku(String code, int han, int yakuman) {}
}
