package com.epsilon.reviewer.engine.scenario;

/**
 * もしもの 1 判断。
 *
 * @param round 解析結果の rounds の位置
 * @param eventIndex 判断の resolve（解かれた出来事の位置）
 * @param seat 判断の席
 * @param call 応答の判断で鳴いたとするチー・ポンの行動 ID。鳴かなければ null
 */
public record ScenarioQuery(
    int round, int eventIndex, int seat, ScenarioEdits edits, Integer call) {}
