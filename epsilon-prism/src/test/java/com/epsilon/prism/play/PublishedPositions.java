package com.epsilon.prism.play;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.prism.TableWalk;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.EnumSet;
import java.util.HexFormat;

/** 配布モデルの照合に使う、親の第一打と二つの半荘から等間隔に選んだ複数択の判断。 */
public final class PublishedPositions {
  private enum History {
    LONG_RIVER,
    OPPONENT_CALL,
    OPPONENT_RIICHI,
    KAN,
    NUKI
  }

  /** 河が長いとみなす、1 席あたりの平均打牌数の下限。 */
  private static final int LONG_RIVER_DISCARDS_PER_SEAT = 12;

  private PublishedPositions() {}

  /** 手番・応答や長さの異なる河を含む、ルールごとの固定 32 局面。 */
  public static ObservationBatch sample(Variant variant) {
    var selected = new ObservationBatch(variant, 1024);
    Engine engine = new Engine(variant);
    engine.startGame(71, DrawOrder.WALL_ORDER);
    engine.advance(new RuleWorkspace(variant));
    var decisions = engine.decisions();
    new ObservationWriter(variant).capture(engine.view(decisions.seat(0)), decisions, 0, selected);
    sampleGame(selected, TableWalk.Policy.GREEDY, 71, 15);
    sampleGame(selected, TableWalk.Policy.CALLS, 72, 16);
    requireHistories(selected);
    return selected;
  }

  private static void sampleGame(
      ObservationBatch selected, TableWalk.Policy policy, long seed, int count) {
    Variant variant = selected.variant();
    var game = new ObservationBatch(variant, 4096);
    var writer = new ObservationWriter(variant);
    new TableWalk(variant, policy)
        .run(
            seed,
            1,
            (engine, slot) -> {
              var decisions = engine.decisions();
              if (decisions.legalCount(slot) > 1) {
                writer.capture(engine.view(decisions.seat(slot)), decisions, slot, game);
              }
            });
    if (game.size() <= count)
      throw new AssertionError("Too few decisions in " + policy + " game: " + game.size());
    var row = new Observation(variant);
    for (int i = 1; i <= count; i++) {
      selected.add(row.bind(game, i * game.size() / (count + 1)));
    }
  }

  private static void requireHistories(ObservationBatch rows) {
    var missing = EnumSet.allOf(History.class);
    if (rows.variant() == Variant.TENHOU_4P) missing.remove(History.NUKI);
    var row = new Observation(rows.variant());
    for (int i = 0; i < rows.size(); i++) {
      row.bind(rows, i);
      for (int seat = 1; seat < rows.variant().seats(); seat++) {
        if (row.riichi(seat) != RiichiState.NONE) missing.remove(History.OPPONENT_RIICHI);
      }
      int discards = 0;
      for (int event = 0; event < row.events(); event++) {
        switch (row.eventKind(event)) {
          case DISCARD -> discards++;
          case NUKI -> missing.remove(History.NUKI);
          case MELD -> {
            var type = row.meld(event).type();
            if (type.kan()) missing.remove(History.KAN);
            if (row.eventAuthor(event) != 0 && !type.concealed())
              missing.remove(History.OPPONENT_CALL);
          }
        }
      }
      if (discards >= LONG_RIVER_DISCARDS_PER_SEAT * rows.variant().seats())
        missing.remove(History.LONG_RIVER);
    }
    if (!missing.isEmpty())
      throw new AssertionError("Missing public histories in " + rows.variant() + ": " + missing);
  }

  /** 各判断のワード数（32 ビット）と観測の全ワード（64 ビット）を順に並べたビッグエンディアン列の SHA-256。 */
  static String fingerprint(ObservationBatch rows) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      for (int i = 0; i < rows.size(); i++) {
        int words = rows.longs(i);
        var bytes = ByteBuffer.allocate(Integer.BYTES + Long.BYTES * words).putInt(words);
        bytes.asLongBuffer().put(rows.words(), rows.start(i), words);
        digest.update(bytes.array());
      }
      return HexFormat.of().formatHex(digest.digest());
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError(e);
    }
  }
}
