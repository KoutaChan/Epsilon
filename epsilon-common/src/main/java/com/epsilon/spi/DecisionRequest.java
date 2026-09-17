package com.epsilon.spi;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineDecisionKind;
import java.util.List;

/** 停止中のエンジンから借用する判断要求。推論バッチが完了するまで有効で、呼び出し元は変更したり完了後も保持したりしてはならない。 */
public record DecisionRequest(
    long id,
    int player,
    EngineDecisionKind kind,
    PublicObservation observation,
    List<Action> legalActions) {}
