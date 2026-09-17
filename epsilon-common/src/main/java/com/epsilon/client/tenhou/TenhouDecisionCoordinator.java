package com.epsilon.client.tenhou;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.ActionGenerator;
import com.epsilon.engine.EngineActionBuffer;
import com.epsilon.engine.Player;
import com.epsilon.engine.WinLegality.RonStatus;
import java.util.List;

/** 合法手生成から天鳳メッセージ生成まで、自家の意思決定だけを担当する。 */
final class TenhouDecisionCoordinator {

  private static final int SELF = 0;

  static final class ResponsePlan {
    private List<Action> actions = List.of();
    private RonStatus ronStatus;

    ResponsePlan bind(List<Action> actions, RonStatus ronStatus) {
      this.actions = actions;
      this.ronStatus = ronStatus;
      return this;
    }

    List<Action> actions() {
      return actions;
    }

    RonStatus ronStatus() {
      return ronStatus;
    }
  }

  private final Player player;
  private final EngineActionBuffer turnActions = new EngineActionBuffer();
  private final EngineActionBuffer callDahaiActions = new EngineActionBuffer();
  private final EngineActionBuffer responseActions = new EngineActionBuffer();
  private final ActionGenerator actionGenerator = new ActionGenerator();
  private final ResponsePlan responsePlan = new ResponsePlan();

  TenhouDecisionCoordinator(Player player) {
    this.player = player;
  }

  List<String> respond(TenhouRoundOutcome outcome, TenhouRoundState round) {
    return switch (outcome) {
      case TenhouRoundOutcome.None ignored -> List.of();
      case TenhouRoundOutcome.DrawDecision draw ->
          selectAndEncode(
              prepareDrawActions(round.state(), draw, turnActions, actionGenerator), round);
      case TenhouRoundOutcome.PostCallDahai postCall -> {
        ActionGenerator.generateCallDahaiActionsInto(
            callDahaiActions, round.state().hand(SELF), postCall.restriction());
        yield selectAndEncode(callDahaiActions, round);
      }
      case TenhouRoundOutcome.ResponseDecision response -> respondTo(response, round);
    };
  }

  private List<String> respondTo(
      TenhouRoundOutcome.ResponseDecision response, TenhouRoundState round) {
    ResponsePlan plan =
        prepareResponse(round.state(), response, responseActions, responsePlan, actionGenerator);
    round.observeResponse(plan.ronStatus());
    if (plan.actions().isEmpty()) {
      return List.of();
    }

    Action action = player.selectAction(round.state(), SELF, plan.actions());
    round.completeResponse(response, plan.ronStatus(), action);
    return encode(action, round);
  }

  private List<String> selectAndEncode(List<Action> actions, TenhouRoundState round) {
    if (actions.isEmpty()) {
      return List.of();
    }
    return encode(player.selectAction(round.state(), SELF, actions), round);
  }

  private static List<String> encode(Action action, TenhouRoundState round) {
    return List.of(
        TenhouActionEncoder.encodeAction(action, round.selfTiles(), round.state().getTurnEvent()));
  }

  private static List<Action> prepareDrawActions(
      GameState state,
      TenhouRoundOutcome.DrawDecision decision,
      EngineActionBuffer destination,
      ActionGenerator actionGenerator) {
    actionGenerator.generateTurnActionsInto(destination, state, SELF, decision.draw());
    retainPromptActions(destination, decision.prompt());
    return destination;
  }

  static ResponsePlan prepareResponse(
      GameState state,
      TenhouRoundOutcome.ResponseDecision response,
      EngineActionBuffer destination,
      ResponsePlan plan,
      ActionGenerator actionGenerator) {
    destination.clear();
    if (!response.prompt().present()) {
      return plan.bind(destination, actionGenerator.ronStatus(state, SELF, response.source()));
    }

    RonStatus ronStatus =
        actionGenerator.generateResponseActionsInto(destination, state, SELF, response.source());
    retainPromptActions(destination, response.prompt());
    return plan.bind(destination, ronStatus);
  }

  private static void retainPromptActions(List<Action> actions, TenhouEvent.ActionPrompt prompt) {
    actions.removeIf(action -> !promptAllows(action.type(), prompt));
  }

  private static boolean promptAllows(Action.Type type, TenhouEvent.ActionPrompt prompt) {
    return switch (type) {
      case DAHAI -> prompt.canDahai();
      case RIICHI_DAHAI -> prompt.canRiichi();
      case TSUMO_AGARI -> prompt.canTsumo();
      case RON_AGARI -> prompt.canRon();
      case CHI -> prompt.canChi();
      case PON -> prompt.canPon();
      case ANKAN, KAKAN, DAIMINKAN -> prompt.canKan();
      case KYUSHU_KYUHAI -> prompt.canKyushu();
      case PASS -> prompt.phase() == TenhouEvent.PromptPhase.RESPONSE;
    };
  }
}
