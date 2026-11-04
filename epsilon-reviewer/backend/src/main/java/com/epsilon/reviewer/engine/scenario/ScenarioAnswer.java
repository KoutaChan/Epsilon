package com.epsilon.reviewer.engine.scenario;

import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.epsilon.reviewer.engine.view.TableState;

/**
 * もしもの答え。候補は解析の判断と同じ形で、chosen は牌譜の手がまだ合法ならその位置。
 *
 * @param table 鳴いた後の局面。鳴かない時は null（画面が変更から作る）
 */
public record ScenarioAnswer(EvaluatedDecision decision, TableState table) {}
