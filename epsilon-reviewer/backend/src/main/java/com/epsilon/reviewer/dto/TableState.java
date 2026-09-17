package com.epsilon.reviewer.dto;

import java.util.List;

/** 差分を適用する起点として保存する、全席の表示状態。作成後は変更しない。 */
public record TableState(
    List<PlayerState> players,
    List<String> dora,
    int honba,
    int kyotaku,
    int remaining,
    int activeSeat) {}
