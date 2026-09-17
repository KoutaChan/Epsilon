package com.epsilon.client.riichi;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.Player;
import com.epsilon.engine.WinLegality.RonStatus;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 一つの RiichiLab 接続で、差分観測・意思決定・応答受領通知を順に処理する。 */
public final class RiichiSession {
  private static final Logger log = LoggerFactory.getLogger(RiichiSession.class);
  private final Player player;
  private final boolean validation;
  private final RiichiActionCandidates candidates = new RiichiActionCandidates();
  private MjaiRoundState game;
  private int seat = -1;
  private boolean completed;
  private Boolean validationPassed;
  private long requestId = -1;
  private Action submittedAction;
  private boolean offeredRon;
  private Action pendingRiichiDiscard;
  private boolean reachAccepted;

  /** プレイヤーを借用して、一接続分の観測と応答を管理する。 */
  public RiichiSession(Player player, boolean validation) {
    this.player = player;
    this.validation = validation;
  }

  /** 完成したテキストフレームを処理し、今回送る一件の応答だけを返す。 */
  public String handleMessage(JsonObject message, long receivedNanos) {
    String type = message.has("type") ? message.get("type").getAsString() : "";
    switch (type) {
      case "start_game" -> assignSeat(message.get("id").getAsInt());
      case "request_action" -> {
        try {
          return respond(message, receivedNanos);
        } catch (RuntimeException error) {
          throw new IllegalArgumentException("Cannot answer Riichi request " + requestId, error);
        }
      }
      case "action_ack" -> acknowledge(message);
      // 終局後は次の要求が来ない場合もある。結果通知は記録だけ行い、差分を二重適用しない。
      case "hora", "ryukyoku" -> log.info("Riichi round result: {}", message);
      case "end_kyoku" -> log.info("Riichi round completed: {}", message);
      case "end_game" -> {
        completed = true;
        pendingRiichiDiscard = null;
        log.info("Riichi game completed: scores={}", message.get("scores"));
      }
      case "validation_result" -> {
        validationPassed = message.get("passed").getAsBoolean();
        log.info(
            "Riichi validation: passed={} reason={}",
            validationPassed,
            message.has("reason") ? message.get("reason").getAsString() : "");
      }
      default -> {
        /* 公開イベントは observation.events からだけ反映する。 */
      }
    }
    return null;
  }

  public boolean started() {
    return seat >= 0;
  }

  public boolean completed() {
    return completed;
  }

  public boolean ended() {
    return validation ? validationPassed != null : completed;
  }

  public boolean validationPassed() {
    return Boolean.TRUE.equals(validationPassed);
  }

  private void assignSeat(int playerId) {
    if (playerId < 0 || playerId >= GameState.NUM_PLAYERS || (seat >= 0 && seat != playerId)) {
      throw new IllegalArgumentException("Invalid Riichi seat: " + playerId);
    }
    if (seat < 0) {
      seat = playerId;
      game = new MjaiRoundState(playerId);
      log.info("Riichi game started: mode={} self=P{}", validation ? "validation" : "ranked", seat);
    }
  }

  private String respond(JsonObject message, long receivedNanos) {
    long incomingRequestId = message.get("request_id").getAsLong();
    if (incomingRequestId <= requestId) return null;
    requestId = incomingRequestId;
    submittedAction = null;
    byte[] decoded = Base64.getDecoder().decode(message.get("observation").getAsString());
    JsonObject observation =
        JsonParser.parseString(new String(decoded, StandardCharsets.UTF_8)).getAsJsonObject();
    assignSeat(observation.get("player_id").getAsInt());
    int previousRound = game.roundNumber();
    game.apply(observation.getAsJsonArray("events"));
    if (observation.has("scores")) {
      game.round().applyScores(MjaiRoundState.integers(observation.getAsJsonArray("scores")));
    }
    if (observation.has("riichi_sticks")) {
      game.round().state().setKyotakuCount(observation.get("riichi_sticks").getAsInt());
    }
    if (game.roundNumber() != previousRound) {
      pendingRiichiDiscard = null;
      reachAccepted = false;
    }
    if (game.roundNumber() == 0)
      throw new IllegalArgumentException("Missing start_kyoku observation");
    candidates.bind(observation, message.getAsJsonArray("possible_actions"), game.round());
    offeredRon = candidates.actions().contains(Action.ronAgari());
    long deadlineMillis = message.getAsJsonObject("time").get("deadline_ms").getAsLong();
    if (expired(receivedNanos, deadlineMillis)) return null;
    Action selected;
    if (pendingRiichiDiscard != null) {
      if (!reachAccepted && !game.selfReachDeclared()) {
        throw new IllegalArgumentException(
            "Riichi follow-up arrived before declaration was accepted");
      }
      selected =
          Action.dahai(pendingRiichiDiscard.tileType(), pendingRiichiDiscard.tileSelection());
    } else selected = player.selectAction(game.round().state(), seat, candidates.actions());
    if (expired(receivedNanos, deadlineMillis)) return null;
    String response = candidates.encode(selected, requestId, game.round());
    submittedAction = selected;
    if (selected.type() == Action.Type.RIICHI_DAHAI) {
      pendingRiichiDiscard = selected;
      reachAccepted = false;
    } else pendingRiichiDiscard = null;
    return response;
  }

  private void acknowledge(JsonObject message) {
    long acknowledgedId = message.get("request_id").getAsLong();
    String status = message.get("status").getAsString();
    log.info("Riichi action receipt: {}", message);
    if (status.equals("rejected") || status.equals("unparseable")) {
      pendingRiichiDiscard = null;
      throw new IllegalArgumentException(
          "Riichi action "
              + status
              + ": request="
              + acknowledgedId
              + " reason="
              + (message.has("reason") ? message.get("reason").getAsString() : ""));
    }
    if (acknowledgedId != requestId || game == null) return;
    if (status.equals("accepted")) {
      if (submittedAction != null) {
        game.round().completeResponse(RonStatus.AVAILABLE, offeredRon, submittedAction);
        if (submittedAction.type() == Action.Type.RIICHI_DAHAI) reachAccepted = true;
      }
    } else if (status.equals("defaulted")) {
      game.round().completeResponse(RonStatus.AVAILABLE, offeredRon, Action.pass());
      pendingRiichiDiscard = null;
      log.warn("Riichi response defaulted: request={}", acknowledgedId);
    } else if (status.equals("stale")) pendingRiichiDiscard = null;
  }

  private static boolean expired(long receivedNanos, long deadlineMillis) {
    return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - receivedNanos) >= deadlineMillis;
  }
}
