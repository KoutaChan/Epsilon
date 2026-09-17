package com.epsilon.replay.format;

import com.epsilon.core.Tile;
import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent.*;
import com.epsilon.replay.format.MahjongSoulRecord.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** 雀魂の元牌譜を、中間の別形式を作らず共通の対局データへ復号する。 */
public final class MahjongSoulReader {
  private static final String[] HONORS = {"E", "S", "W", "N", "P", "F", "C"};
  private final List<ReplayEvent> events = new ArrayList<>();
  private final TileIdentityAllocator tiles = new TileIdentityAllocator();
  private final List<String> indicators = new ArrayList<>();
  private final boolean[] acceptedRiichi = new boolean[4];
  private final List<String> names = new ArrayList<>(List.of("", "", "", ""));
  private int[] scores;
  private int[] finalScores;
  private String finalScoresSource = "unavailable";
  private boolean inRound;
  private boolean completed;
  private int discardSeat = -1;
  private int initialDealer;
  private int roundWind;
  private int initialDealerDraw = -1;

  private MahjongSoulReader() {
    events.add(new StartGame());
  }

  /** 圧縮解除済みの JSON または Liqi protobuf を読み込む。 */
  public static ReplayRecord readRecord(byte[] payload) {
    if (payload.length == 0) throw new IllegalArgumentException("Mahjong Soul record is empty.");
    var reader = new MahjongSoulReader();
    String prefix =
        new String(payload, 0, Math.min(payload.length, 128), StandardCharsets.UTF_8).strip();
    if (prefix.startsWith("\uFEFF")) prefix = prefix.substring(1).stripLeading();
    if (prefix.startsWith("{") || prefix.startsWith("["))
      MahjongSoulJson.readSourceRecords(payload, reader);
    else MahjongSoulBinary.readSourceRecords(payload, reader);
    if (reader.events.size() == 1)
      throw new IllegalArgumentException("Record is missing its initial round.");
    if (reader.finalScores != null && !reader.inRound) reader.completed = true;
    if (reader.completed) reader.events.add(new EndGame(reader.finalScores));
    return new ReplayRecord(
        reader.events,
        new RecordMetadata(
            RecordFormat.MAHJONG_SOUL,
            reader.names,
            reader.finalScores,
            reader.finalScoresSource,
            null),
        null,
        reader.completed ? RecordCompletion.COMPLETE : RecordCompletion.INCOMPLETE);
  }

  void readPlayerName(int seat, String name) {
    requireSeat(seat);
    names.set(seat, name);
  }

  void readFinalScores(int[] values, String source) {
    requireFourScores(values);
    finalScores = values;
    finalScoresSource = source;
  }

  void acceptRecord(MahjongSoulRecord record) {
    if (completed) throw new IllegalArgumentException("Record contains events after game end.");
    if (record instanceof Round round) {
      startRound(round);
      return;
    }
    if (!inRound) throw new IllegalArgumentException("Event occurs without an active round.");
    switch (record) {
      case Draw draw -> {
        requireSeat(draw.seat());
        acceptRiichi(draw.riichi());
        initialDealerDraw = -1;
        revealIndicators(draw.indicators());
        int id =
            tiles.allocateDrawTile(
                draw.seat(),
                TileNotation.parseTileType(draw.tile()),
                TileNotation.isRed(draw.tile()));
        events.add(new Tsumo(draw.seat(), id));
      }
      case Discard discard -> discardTile(discard);
      case Call call -> {
        acceptRiichi(call.riichi());
        callTiles(call);
      }
      case Kan kan -> callKan(kan);
      case Win win -> recordWin(win);
      case ExhaustiveDraw draw -> {
        applyDeltas(draw.deltas());
        events.add(new Ryukyoku(draw.deltas(), "EXHAUSTIVE_DRAW"));
        finishRound(draw.end());
      }
      case AbortiveDraw draw -> {
        acceptRiichi(draw.riichi());
        events.add(
            new Ryukyoku(
                new int[4],
                draw.reason() == 1 ? "NINE_TERMINALS" : "MAHJONG_SOUL_" + draw.reason()));
        finishRound(draw.end());
      }
      default -> throw new IllegalArgumentException("Unexpected Mahjong Soul round event.");
    }
  }

  private void startRound(Round round) {
    if (inRound) throw new IllegalArgumentException("Round is missing its terminal event.");
    requireSeat(round.dealer());
    if (round.wind() < 0 || round.wind() > 3 || round.honba() < 0 || round.deposits() < 0)
      throw new IllegalArgumentException("Invalid Mahjong Soul round metadata.");
    requireFourScores(round.scores());
    if (round.hands().size() != 4 || round.indicators().isEmpty())
      throw new IllegalArgumentException("Initial hands and dora indicator are required.");
    scores = round.scores().clone();
    indicators.clear();
    indicators.add(canonicalTile(round.indicators().getFirst()));
    Arrays.fill(acceptedRiichi, false);
    discardSeat = -1;
    initialDealer = round.dealer();
    roundWind = round.wind();
    initialDealerDraw = -1;
    String indicator = indicators.getFirst();
    tiles.startRound(TileNotation.parseTileType(indicator), TileNotation.isRed(indicator));
    int[][] hands = new int[4][];
    for (int seat = 0; seat < 4; seat++) {
      List<String> hand = round.hands().get(seat);
      if (hand.size() != 13 && !(seat == initialDealer && hand.size() == 14))
        throw new IllegalArgumentException(
            "Four-player initial hands must contain all tile identities.");
      hands[seat] = new int[13];
      for (int i = 0; i < 13; i++)
        hands[seat][i] =
            tiles.allocateInitialTile(
                seat, TileNotation.parseTileType(hand.get(i)), TileNotation.isRed(hand.get(i)));
    }
    events.add(
        new StartKyoku(
            Tile.TON + roundWind,
            initialDealer + 1,
            round.honba(),
            round.deposits(),
            initialDealer,
            TileNotation.parseTileType(indicator),
            hands,
            round.scores(),
            TileNotation.isRed(indicator)));
    inRound = true;
    revealIndicators(round.indicators());
    normalizeInitialDealerDraw(round.hands().get(initialDealer));
  }

  /** 雀魂の親配牌14枚の末尾を、共通再生の13枚配牌と初回ツモへ分ける。 */
  private void normalizeInitialDealerDraw(List<String> dealerHand) {
    if (dealerHand.size() == 14) {
      String tile = dealerHand.get(13);
      initialDealerDraw =
          tiles.allocateDrawTile(
              initialDealer, TileNotation.parseTileType(tile), TileNotation.isRed(tile));
      events.add(new Tsumo(initialDealer, initialDealerDraw, true));
    }
  }

  private void discardTile(Discard discard) {
    requireSeat(discard.seat());
    if (discard.riichi()) events.add(new Reach(discard.seat()));
    int tileType = TileNotation.parseTileType(discard.tile());
    boolean tileRed = TileNotation.isRed(discard.tile());
    // 初回14枚にはツモ切り区分がない。独立させた唯一の同一牌を捨てる場合だけ正規化する。
    boolean initialDiscard =
        discard.seat() == initialDealer
            && initialDealerDraw >= 0
            && Tile.typeOf(initialDealerDraw) == tileType
            && Tile.isAka(initialDealerDraw) == tileRed
            && Arrays.stream(tiles.handTilesOfType(discard.seat(), tileType))
                    .filter(id -> Tile.isAka(id) == tileRed)
                    .count()
                == 1;
    boolean tsumogiri = discard.tsumogiri() || initialDiscard;
    int id = tiles.removeDiscardTile(discard.seat(), tileType, tileRed, tsumogiri);
    events.add(new Dahai(discard.seat(), id, tsumogiri, initialDiscard && !discard.tsumogiri()));
    initialDealerDraw = -1;
    discardSeat = discard.seat();
    revealIndicators(discard.indicators());
  }

  private void acceptRiichi(Riichi riichi) {
    if (riichi == null || riichi.failed()) return;
    requireSeat(riichi.seat());
    if (!acceptedRiichi[riichi.seat()]) {
      acceptedRiichi[riichi.seat()] = true;
      scores[riichi.seat()] -= 1000;
      events.add(new ReachAccepted(riichi.seat()));
    }
  }

  private void callTiles(Call call) {
    requireSeat(call.seat());
    int count = call.type() == 2 ? 4 : 3;
    if (call.type() < 0
        || call.type() > 2
        || call.tiles().size() != count
        || call.sources().length != count) throw new IllegalArgumentException("Invalid meld data.");
    int[] consumed = new int[count - 1];
    int consumedCount = 0, calledId = -1, target = -1;
    for (int i = 0; i < count; i++) {
      String tile = call.tiles().get(i);
      int from = call.sources()[i];
      if (from == call.seat()) {
        if (consumedCount == consumed.length)
          throw new IllegalArgumentException("Meld has no called tile.");
        consumed[consumedCount++] = TileNotation.representativeTileId(tile);
      } else {
        if (calledId >= 0 || from != discardSeat)
          throw new IllegalArgumentException("Invalid called-tile source.");
        target = from;
        calledId =
            tiles.takeCalledTile(from, TileNotation.parseTileType(tile), TileNotation.isRed(tile));
      }
    }
    if (calledId < 0 || consumedCount != consumed.length)
      throw new IllegalArgumentException("Meld is missing its called tile.");
    consumed = tiles.consumeMeldTiles(call.seat(), consumed);
    events.add(
        switch (call.type()) {
          case 0 -> new Chi(call.seat(), target, calledId, consumed);
          case 1 -> new Pon(call.seat(), target, calledId, consumed);
          case 2 -> new Daiminkan(call.seat(), target, calledId, consumed);
          default -> throw new IllegalArgumentException("Invalid meld type.");
        });
  }

  private void callKan(Kan kan) {
    requireSeat(kan.seat());
    int[] owned = tiles.handTilesOfType(kan.seat(), TileNotation.parseTileType(kan.tile()));
    if (kan.type() == 3 && owned.length == 4) {
      events.add(new Ankan(kan.seat(), tiles.consumeMeldTiles(kan.seat(), owned)));
    } else if (kan.type() == 2 && owned.length == 1) {
      events.add(new Kakan(kan.seat(), tiles.consumeMeldTiles(kan.seat(), owned)[0]));
    } else throw new IllegalArgumentException("Invalid kan tile data.");
    discardSeat = kan.seat();
    initialDealerDraw = -1;
    revealIndicators(kan.indicators());
  }

  private void recordWin(Win win) {
    requireFourScores(win.deltas());
    if (win.winners().isEmpty()) throw new IllegalArgumentException("Winning event has no winner.");
    boolean first = true;
    for (Winner winner : win.winners()) {
      requireSeat(winner.seat());
      int target = winner.tsumo() ? winner.seat() : discardSeat;
      if (target < 0) throw new IllegalArgumentException("Discarding player is unknown.");
      events.add(
          new Hora(
              winner.seat(),
              target,
              TileNotation.parseTileType(winner.tile()),
              first ? win.deltas() : new int[4],
              winDetails(winner)));
      first = false;
    }
    applyDeltas(win.deltas());
    if (win.scores() != null && !Arrays.equals(scores, win.scores()))
      throw new IllegalArgumentException("Scores after the win do not match.");
    finishRound(win.end());
  }

  private WinDetails winDetails(Winner winner) {
    List<WinDetails.Yaku> yaku = null;
    if (winner.fans() != null) {
      yaku = new ArrayList<>();
      for (Fan fan : winner.fans()) {
        if (fan.value() > 0)
          yaku.add(
              new WinDetails.Yaku(
                  yakuCode(fan.id(), winner.seat()),
                  winner.yakuman() ? 0 : fan.value(),
                  winner.yakuman() ? fan.value() : 0));
      }
    }
    List<String> ura =
        winner.uraIndicators() == null
            ? null
            : winner.uraIndicators().stream().map(MahjongSoulReader::canonicalTile).toList();
    if (ura == null && Boolean.FALSE.equals(winner.riichi())) ura = List.of();
    Integer points = winner.ronPoints();
    if (winner.tsumo())
      points =
          winner.childPayment() == null
              ? null
              : winner.seat() == initialDealer
                  ? Math.multiplyExact(winner.childPayment(), 3)
                  : winner.dealerPayment() == null
                      ? null
                      : Math.addExact(
                          winner.dealerPayment(), Math.multiplyExact(winner.childPayment(), 2));
    // point_sum は対局中の累計であり、和了点には使用しない。
    return new WinDetails(
        winner.yakuman() ? null : winner.count(),
        winner.fu(),
        points,
        winner.yakuman() ? winner.count() : winner.count() == null ? null : 0,
        yaku,
        ura);
  }

  private void finishRound(GameEnd end) {
    if (end.complete()) {
      completed = true;
      if (end.scores() != null) readFinalScores(end.scores(), "majsoul.gameend.scores");
      else readFinalScores(scores, "majsoul.gameend.terminal_deltas");
    }
    events.add(new EndKyoku());
    inRound = false;
  }

  private void revealIndicators(List<String> current) {
    if (current.isEmpty()) return;
    if (current.size() > 5 || current.size() < indicators.size())
      throw new IllegalArgumentException("Invalid dora indicator count.");
    for (int i = 0; i < current.size(); i++) {
      String marker = canonicalTile(current.get(i));
      if (i < indicators.size()) {
        if (!indicators.get(i).equals(marker))
          throw new IllegalArgumentException("Previously revealed dora indicator changed.");
      } else {
        indicators.add(marker);
        tiles.reserveIndicator(TileNotation.parseTileType(marker), TileNotation.isRed(marker));
        events.add(new Dora(TileNotation.parseTileType(marker), TileNotation.isRed(marker)));
      }
    }
  }

  private void applyDeltas(int[] deltas) {
    requireFourScores(deltas);
    for (int seat = 0; seat < 4; seat++) scores[seat] = Math.addExact(scores[seat], deltas[seat]);
  }

  static void requireFourScores(int[] values) {
    if (values.length != 4)
      throw new IllegalArgumentException("Scores for all four players are required.");
  }

  private static void requireSeat(int seat) {
    if (seat < 0 || seat >= 4)
      throw new IllegalArgumentException("Seat must identify one of four players.");
  }

  private static String canonicalTile(String tile) {
    return TileNotation.formatPhysicalTile(TileNotation.representativeTileId(tile));
  }

  private String yakuCode(int id, int seat) {
    return switch (id) {
      case 1 -> "MENZEN_TSUMO";
      case 2 -> "RIICHI";
      case 3 -> "CHANKAN";
      case 4 -> "RINSHAN";
      case 5 -> "HAITEI";
      case 6 -> "HOUTEI";
      case 7 -> "YAKUHAI_HAKU";
      case 8 -> "YAKUHAI_HATSU";
      case 9 -> "YAKUHAI_CHUN";
      case 10 -> "YAKUHAI_SEAT_" + HONORS[(seat - initialDealer + 4) % 4];
      case 11 -> "YAKUHAI_ROUND_" + HONORS[roundWind];
      case 12 -> "TANYAO";
      case 13 -> "IPEIKOU";
      case 14 -> "PINFU";
      case 15 -> "CHANTA";
      case 16 -> "ITTSU";
      case 17 -> "SANSHOKU";
      case 18 -> "DOUBLE_RIICHI";
      case 19 -> "SANSHOKU_DOUKOU";
      case 20 -> "SANKANTSU";
      case 21 -> "TOITOI";
      case 22 -> "SANANKOU";
      case 23 -> "SHOUSANGEN";
      case 24 -> "HONROUTOU";
      case 25 -> "CHIITOITSU";
      case 26 -> "JUNCHAN";
      case 27 -> "HONITSU";
      case 28 -> "RYANPEIKOU";
      case 29 -> "CHINITSU";
      case 30 -> "IPPATSU";
      case 31 -> "DORA";
      case 32 -> "AKADORA";
      case 33 -> "URADORA";
      case 34 -> "NUKIDORA";
      case 35 -> "TENHOU";
      case 36 -> "CHIIHOU";
      case 37 -> "DAISANGEN";
      case 38 -> "SUUANKOU";
      case 39 -> "TSUUIISOU";
      case 40 -> "RYUUIISOU";
      case 41 -> "CHINROUTOU";
      case 42 -> "KOKUSHI";
      case 43 -> "SHOUSUUSHII";
      case 44 -> "SUUKANTSU";
      case 45 -> "CHUUREN";
      case 47 -> "JUNSEI_CHUUREN";
      case 48 -> "SUUANKOU_TANKI";
      case 49 -> "KOKUSHI_13";
      case 50 -> "DAISUUSHII";
      case 59, 65 -> "RENHOU";
      default -> "MAJSOUL_" + id;
    };
  }
}
