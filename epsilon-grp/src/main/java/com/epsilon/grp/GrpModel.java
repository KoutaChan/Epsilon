package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;

/**
 * Φ を出すモデル。局末の卓だけを読む GrpNetwork と、各局が始まったときの卓の履歴（RoundHistory）も読む GruSinkhorn がある。
 * 重みは書き換えない。推論は純 Java で、スレッド・メモ・呼び出しの順に依らずビット単位で同じ値を出す。使う側は型で処理を
 * 分ける（Phi と GrpCheckpoint の switch）。
 */
public sealed interface GrpModel permits GrpNetwork, GruSinkhorn {

  Variant variant();
}
