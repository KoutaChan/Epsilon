package com.epsilon.reviewer.dto;

/** 指定席の表示状態の変更。nullの項目は直前の値を引き継ぐ。 */
public record PlayerDelta(
    int seat,
    Integer score,
    Boolean riichi,
    ArrayPatch<String> hand,
    DrawUpdate draw,
    ArrayPatch<RiverTile> river,
    ArrayPatch<MeldSnapshot> melds) {}
