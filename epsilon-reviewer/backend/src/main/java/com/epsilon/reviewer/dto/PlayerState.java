package com.epsilon.reviewer.dto;

import java.util.List;

/** 一局面におけるプレイヤーの表示状態。席は配列の位置で表し、名前は対局の付加情報で保持する。 */
public record PlayerState(
    int score,
    boolean riichi,
    List<String> hand,
    String draw,
    List<RiverTile> river,
    List<MeldSnapshot> melds) {}
