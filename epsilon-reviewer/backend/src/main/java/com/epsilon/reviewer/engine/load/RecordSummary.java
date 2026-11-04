package com.epsilon.reviewer.engine.load;

import java.util.List;

/**
 * 牌譜の概要。読み込み・履歴・解析結果で同じ形。
 *
 * @param source 牌譜の形式（tenhou・mjai・majsoul）
 * @param names 4 席の名前。名前の無い席は空の文字列
 * @param finalScores 4 席の最終の持ち点。牌譜から分からなければ null
 * @param roundCount 局の数
 */
public record RecordSummary(
    String source, List<String> names, List<Integer> finalScores, int roundCount) {}
