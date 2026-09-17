package com.epsilon.replay.format;

import com.epsilon.client.tenhou.TenhouMeldDecoder;
import com.epsilon.client.tenhou.TenhouWallDecoder;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent.*;
import java.io.StringReader;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.xml.stream.*;

/** 天鳳 XML の属性・名前・採点・イベントを StAX の一走査で読み込む。 */
public final class TenhouXmlReader {
  private static final String DRAW_LETTERS = "TUVW";
  private static final String DISCARD_LETTERS = "DEFG";
  private final List<ReplayEvent> events = new ArrayList<>();
  private final List<String> names = new ArrayList<>(List.of("", "", "", ""));
  private final int[] drawn = {-1, -1, -1, -1};
  private String shuffleSeed;
  private String ruleDescription;
  private int[] finalScores;
  private int roundCount;
  private boolean roundSettled;

  private TenhouXmlReader() {
    events.add(new StartGame());
  }

  public static ReplayRecord readRecord(String text) {
    XMLInputFactory factory = XMLInputFactory.newFactory();
    factory.setProperty(XMLInputFactory.SUPPORT_DTD, false);
    factory.setProperty("javax.xml.stream.isSupportingExternalEntities", false);
    TenhouXmlReader reader = new TenhouXmlReader();
    try {
      XMLStreamReader xml = factory.createXMLStreamReader(new StringReader(text));
      try {
        while (xml.hasNext()) {
          int event = xml.next();
          if (event == XMLStreamConstants.DTD)
            throw new IllegalArgumentException("DTD declarations are not allowed in Tenhou XML.");
          if (event == XMLStreamConstants.START_ELEMENT) reader.readElement(xml);
        }
      } finally {
        xml.close();
      }
    } catch (XMLStreamException failure) {
      throw new IllegalArgumentException("Invalid Tenhou XML syntax.", failure);
    }
    if (reader.roundCount == 0)
      throw new IllegalArgumentException("Tenhou record contains no rounds.");
    // EOF は対局終了ではない。局結果があれば局を閉じ、owari がある場合だけ対局を閉じる。
    if (reader.roundSettled) reader.events.add(new EndKyoku());
    if (reader.finalScores != null) reader.events.add(new EndGame(reader.finalScores));
    int[][] walls =
        reader.shuffleSeed == null
            ? null
            : TenhouWallDecoder.decodeWalls(reader.shuffleSeed, reader.roundCount);
    return new ReplayRecord(
        reader.events,
        new RecordMetadata(
            RecordFormat.TENHOU,
            reader.names,
            reader.finalScores,
            reader.finalScores == null ? "unavailable" : "tenhou.owari",
            reader.ruleDescription),
        walls,
        reader.finalScores == null ? RecordCompletion.INCOMPLETE : RecordCompletion.COMPLETE);
  }

  private void readElement(XMLStreamReader xml) {
    String name = xml.getLocalName();
    if (name.length() > 1 && Character.isDigit(name.charAt(1))) {
      int actor = DRAW_LETTERS.indexOf(name.charAt(0));
      if (actor >= 0) {
        int id = requirePhysicalTileId(Integer.parseInt(name.substring(1)));
        drawn[actor] = id;
        events.add(new Tsumo(actor, id));
        return;
      }
      actor = DISCARD_LETTERS.indexOf(name.charAt(0));
      if (actor >= 0) {
        int id = requirePhysicalTileId(Integer.parseInt(name.substring(1)));
        events.add(new Dahai(actor, id, drawn[actor] == id));
        drawn[actor] = -1;
        return;
      }
    }
    switch (name) {
      case "mjloggm", "TAIKYOKU", "BYE", "PROF" -> {}
      case "SHUFFLE" -> shuffleSeed = readRequiredAttribute(xml, "seed");
      case "UN" -> {
        for (int seat = 0; seat < 4; seat++) {
          String value = readAttribute(xml, "n" + seat);
          if (value != null) names.set(seat, URLDecoder.decode(value, StandardCharsets.UTF_8));
        }
      }
      case "GO" -> {
        int type = readRequiredInteger(xml, "type");
        if ((type & 16) != 0)
          throw new IllegalArgumentException("Three-player records are not supported.");
        ruleDescription = "Tenhou type=" + type;
      }
      case "INIT" -> startRound(xml);
      case "N" -> {
        int actor = readSeatIndex(xml, "who");
        appendMeldEvent(events, actor, readRequiredInteger(xml, "m"));
        drawn[actor] = -1;
      }
      case "REACH" -> {
        int actor = readSeatIndex(xml, "who");
        switch (readRequiredInteger(xml, "step")) {
          case 1 -> events.add(new Reach(actor));
          case 2 -> events.add(new ReachAccepted(actor));
          default -> throw new IllegalArgumentException("Invalid Tenhou REACH step.");
        }
      }
      case "DORA" -> {
        int id = requirePhysicalTileId(readRequiredInteger(xml, "hai"));
        events.add(new Dora(Tile.typeOf(id), Tile.isAka(id)));
      }
      case "AGARI", "RYUUKYOKU" -> {
        int[] deltas = parseScoreDeltas(readAttribute(xml, "sc"));
        roundSettled = true;
        String owari = readAttribute(xml, "owari");
        if (owari != null) finalScores = parseFourScores(owari, 2);
        if (name.equals("AGARI")) {
          String machi = readAttribute(xml, "machi");
          events.add(
              new Hora(
                  readSeatIndex(xml, "who"),
                  readSeatIndex(xml, "fromWho"),
                  machi == null ? -1 : Tile.typeOf(requirePhysicalTileId(Integer.parseInt(machi))),
                  deltas,
                  parseWinDetails(xml)));
        } else events.add(new Ryukyoku(deltas, decodeDrawReason(readAttribute(xml, "type"))));
      }
      default -> throw new IllegalArgumentException("Unsupported Tenhou XML element: " + name);
    }
  }

  private void startRound(XMLStreamReader xml) {
    if (finalScores != null)
      throw new IllegalArgumentException("A round occurs after the final result.");
    if (roundSettled) events.add(new EndKyoku());
    int[] seed = parseIntegers(readRequiredAttribute(xml, "seed"));
    if (seed.length != 6)
      throw new IllegalArgumentException("Tenhou INIT seed must contain six values.");
    int marker = requirePhysicalTileId(seed[5]);
    int[][] hands = new int[4][];
    for (int seat = 0; seat < 4; seat++) {
      hands[seat] = parseIntegers(readRequiredAttribute(xml, "hai" + seat));
      if (hands[seat].length != 13)
        throw new IllegalArgumentException("Each initial hand must contain 13 tiles.");
      for (int id : hands[seat]) requirePhysicalTileId(id);
    }
    events.add(
        new StartKyoku(
            Tile.TON + seed[0] / 4,
            seed[0] % 4 + 1,
            seed[1],
            seed[2],
            readSeatIndex(xml, "oya"),
            Tile.typeOf(marker),
            hands,
            parseFourScores(readRequiredAttribute(xml, "ten"), 1),
            Tile.isAka(marker)));
    Arrays.fill(drawn, -1);
    roundCount++;
    roundSettled = false;
  }

  private static String decodeDrawReason(String value) {
    if (value == null) return "EXHAUSTIVE_DRAW";
    return switch (value) {
      case "yao9" -> "NINE_TERMINALS";
      case "reach4" -> "FOUR_RIICHI";
      case "ron3" -> "THREE_RON";
      case "kan4" -> "FOUR_KANS";
      case "kaze4" -> "FOUR_WINDS";
      default -> "TENHOU_" + value;
    };
  }

  private static WinDetails parseWinDetails(XMLStreamReader xml) {
    String score = readAttribute(xml, "ten");
    String ordinaryYaku = readAttribute(xml, "yaku");
    String yakumanYaku = readAttribute(xml, "yakuman");
    String ura = readAttribute(xml, "doraHaiUra");
    if (score == null && ordinaryYaku == null && yakumanYaku == null && ura == null) return null;

    Integer fu = null;
    Integer points = null;
    if (score != null) {
      int[] values = parseIntegers(score);
      fu = values[0];
      points = values[1];
    }
    Integer han = null;
    Integer yakuman = null;
    List<WinDetails.Yaku> yaku = null;
    if (yakumanYaku != null) {
      yaku = new ArrayList<>();
      for (int id : parseIntegers(yakumanYaku)) {
        // 天鳳は単騎・十三面・純正なども一倍の役満として記録する。
        yaku.add(new WinDetails.Yaku(decodeYakuCode(id), 0, 1));
      }
      han = 0;
      yakuman = yaku.size();
    } else if (ordinaryYaku != null) {
      int[] values = parseIntegers(ordinaryYaku);
      if (values.length % 2 != 0)
        throw new IllegalArgumentException("Tenhou yaku must contain ID and han pairs.");
      yaku = new ArrayList<>();
      han = 0;
      yakuman = 0;
      for (int i = 0; i < values.length; i += 2) {
        int count = values[i + 1];
        if (count > 0) {
          yaku.add(new WinDetails.Yaku(decodeYakuCode(values[i]), count, 0));
          han += count;
        }
      }
    }
    List<String> uraIndicators = null;
    if (ura != null) {
      uraIndicators = new ArrayList<>();
      for (int physicalId : parseIntegers(ura)) {
        int type = Tile.typeOf(physicalId);
        String tile =
            type < 27
                ? (type % 9 + 1) + "" + "mps".charAt(type / 9)
                : "ESWNPFC".substring(type - 27, type - 26);
        uraIndicators.add(tile + (Tile.isAka(physicalId) ? "r" : ""));
      }
    }
    return new WinDetails(han, fu, points, yakuman, yaku, uraIndicators);
  }

  private static String decodeYakuCode(int id) {
    return switch (id) {
      case 0 -> "MENZEN_TSUMO";
      case 1 -> "RIICHI";
      case 2 -> "IPPATSU";
      case 3 -> "CHANKAN";
      case 4 -> "RINSHAN";
      case 5 -> "HAITEI";
      case 6 -> "HOUTEI";
      case 7 -> "PINFU";
      case 8 -> "TANYAO";
      case 9 -> "IPEIKOU";
      case 10 -> "YAKUHAI_SEAT_E";
      case 11 -> "YAKUHAI_SEAT_S";
      case 12 -> "YAKUHAI_SEAT_W";
      case 13 -> "YAKUHAI_SEAT_N";
      case 14 -> "YAKUHAI_ROUND_E";
      case 15 -> "YAKUHAI_ROUND_S";
      case 16 -> "YAKUHAI_ROUND_W";
      case 17 -> "YAKUHAI_ROUND_N";
      case 18 -> "YAKUHAI_HAKU";
      case 19 -> "YAKUHAI_HATSU";
      case 20 -> "YAKUHAI_CHUN";
      case 21 -> "DOUBLE_RIICHI";
      case 22 -> "CHIITOITSU";
      case 23 -> "CHANTA";
      case 24 -> "ITTSU";
      case 25 -> "SANSHOKU";
      case 26 -> "SANSHOKU_DOUKOU";
      case 27 -> "SANKANTSU";
      case 28 -> "TOITOI";
      case 29 -> "SANANKOU";
      case 30 -> "SHOUSANGEN";
      case 31 -> "HONROUTOU";
      case 32 -> "RYANPEIKOU";
      case 33 -> "JUNCHAN";
      case 34 -> "HONITSU";
      case 35 -> "CHINITSU";
      case 36 -> "RENHOU";
      case 37 -> "TENHOU";
      case 38 -> "CHIIHOU";
      case 39 -> "DAISANGEN";
      case 40 -> "SUUANKOU";
      case 41 -> "SUUANKOU_TANKI";
      case 42 -> "TSUUIISOU";
      case 43 -> "RYUUIISOU";
      case 44 -> "CHINROUTOU";
      case 45 -> "CHUUREN";
      case 46 -> "JUNSEI_CHUUREN";
      case 47 -> "KOKUSHI";
      case 48 -> "KOKUSHI_13";
      case 49 -> "DAISUUSHII";
      case 50 -> "SHOUSUUSHII";
      case 51 -> "SUUKANTSU";
      case 52 -> "DORA";
      case 53 -> "URADORA";
      case 54 -> "AKADORA";
      case 55 -> "NUKIDORA";
      default -> "TENHOU_" + id;
    };
  }

  private static void appendMeldEvent(List<ReplayEvent> events, int who, int m) {
    TenhouMeldDecoder.DecodedMeld decoded = TenhouMeldDecoder.decode(m);
    Meld meld = decoded.meld();
    int[] consumedPhysicalTileIds = decoded.consumedPhysicalTileIds();
    switch (meld.type()) {
      case CHI ->
          events.add(
              new Chi(
                  who,
                  calledFromPlayer(who, meld),
                  decoded.calledPhysicalTileId(),
                  consumedPhysicalTileIds));
      case PON ->
          events.add(
              new Pon(
                  who,
                  calledFromPlayer(who, meld),
                  decoded.calledPhysicalTileId(),
                  consumedPhysicalTileIds));
      case DAIMINKAN ->
          events.add(
              new Daiminkan(
                  who,
                  calledFromPlayer(who, meld),
                  decoded.calledPhysicalTileId(),
                  consumedPhysicalTileIds));
      case ANKAN -> events.add(new Ankan(who, consumedPhysicalTileIds));
      case KAKAN -> events.add(new Kakan(who, consumedPhysicalTileIds[0]));
    }
  }

  private static int calledFromPlayer(int player, Meld meld) {
    return (player + meld.relativeSource().playerOffset()) % 4;
  }

  private static String readAttribute(XMLStreamReader xml, String key) {
    return xml.getAttributeValue(null, key);
  }

  private static String readRequiredAttribute(XMLStreamReader xml, String key) {
    String value = readAttribute(xml, key);
    if (value == null || value.isEmpty())
      throw new IllegalArgumentException("Missing required Tenhou attribute: " + key);
    return value;
  }

  private static int readRequiredInteger(XMLStreamReader xml, String key) {
    return Integer.parseInt(readRequiredAttribute(xml, key));
  }

  private static int readSeatIndex(XMLStreamReader xml, String key) {
    int seat = readRequiredInteger(xml, key);
    if (seat < 0 || seat > 3) throw new IllegalArgumentException("Seat must be between 0 and 3.");
    return seat;
  }

  private static int requirePhysicalTileId(int id) {
    if (id < 0 || id >= 136) throw new IllegalArgumentException("Invalid physical tile ID: " + id);
    return id;
  }

  private static int[] parseFourScores(String value, int stride) {
    String[] fields = value.split(",");
    if (fields.length != 4 * stride)
      throw new IllegalArgumentException("Tenhou scores must contain four players.");
    int[] scores = new int[4];
    for (int i = 0; i < 4; i++)
      scores[i] =
          new BigDecimal(fields[i * stride]).multiply(BigDecimal.valueOf(100)).intValueExact();
    return scores;
  }

  private static int[] parseScoreDeltas(String value) {
    if (value == null) return null;
    String[] fields = value.split(",");
    if (fields.length != 8)
      throw new IllegalArgumentException("Tenhou score deltas must contain four players.");
    int[] deltas = new int[4];
    for (int i = 0; i < 4; i++)
      deltas[i] =
          new BigDecimal(fields[i * 2 + 1]).multiply(BigDecimal.valueOf(100)).intValueExact();
    return deltas;
  }

  private static int[] parseIntegers(String value) {
    if (value == null || value.isEmpty()) return new int[0];
    String[] fields = value.split(",");
    int[] result = new int[fields.length];
    for (int i = 0; i < fields.length; i++) result[i] = Integer.parseInt(fields[i]);
    return result;
  }
}
