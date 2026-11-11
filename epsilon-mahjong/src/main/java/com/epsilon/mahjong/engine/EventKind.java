package com.epsilon.mahjong.engine;

/** 公開イベントの種類。 */
public enum EventKind {
  DISCARD,
  /** チー・ポン・大明槓・暗槓。 */
  MELD,
  /** 加槓。元のポンのイベントの番号を持つ。 */
  KAKAN,
  /** 北抜き。抜いた北は河に入らない。 */
  NUKI
}
