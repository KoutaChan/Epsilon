package com.epsilon.reviewer.dto;

/** 局面の全状態または直前の局面からの差分の一方を保持する。一局内の記録順に適用して状態を復元する。 */
public record StateFrame(int stateId, TableState checkpoint, StateDelta delta) {}
