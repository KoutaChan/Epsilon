package com.epsilon.client.tenhou;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 受信イベントを対局状態へ反映し、天鳳へ返すメッセージを組み立てる。 */
final class TenhouSession {

  private static final Logger log = LoggerFactory.getLogger(TenhouSession.class);

  private final TenhouGameController controller;
  private boolean ended;
  private boolean nextReadySent;

  TenhouSession(TenhouGameController controller) {
    this.controller = controller;
  }

  /** 生メッセージを一度だけ解析して処理する。 */
  List<String> handleMessage(String message) {
    return handleEvent(TenhouMessageParser.parse(message));
  }

  boolean ended() {
    return ended;
  }

  private List<String> handleEvent(TenhouEvent event) {
    if (event instanceof TenhouEvent.GameEvent gameEvent) {
      List<String> responses = new ArrayList<>(controller.handleEvent(gameEvent));
      if (event instanceof TenhouEvent.Init) {
        nextReadySent = false;
      } else if (isGameEnd(event)) {
        ended = true;
      } else if (isRoundEnd(event) && !nextReadySent) {
        responses.add(TenhouActionEncoder.encodeNextReady());
        nextReadySent = true;
      }
      return responses;
    }

    if (event instanceof TenhouEvent.Go) {
      return List.of(TenhouActionEncoder.encodeGok());
    }
    if (event instanceof TenhouEvent.Un un) {
      logPlayerNames(un);
    } else if (event instanceof TenhouEvent.EndGame) {
      ended = true;
    }
    return List.of();
  }

  private static boolean isRoundEnd(TenhouEvent event) {
    return event instanceof TenhouEvent.Agari || event instanceof TenhouEvent.Ryukyoku;
  }

  private static boolean isGameEnd(TenhouEvent event) {
    return (event instanceof TenhouEvent.Agari agari && agari.gameEnd())
        || (event instanceof TenhouEvent.Ryukyoku ryukyoku && ryukyoku.gameEnd());
  }

  private static void logPlayerNames(TenhouEvent.Un event) {
    String[] names = event.names();
    for (int player = 0; player < names.length; player++) {
      String name = names[player];
      if (!name.isEmpty()) {
        log.info("P{}: {}", player, URLDecoder.decode(name, StandardCharsets.UTF_8));
      }
    }
  }
}
