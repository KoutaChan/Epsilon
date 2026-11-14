package com.epsilon.mahjong.io.majsoul;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.record.TileCopies;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.Tiles;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * 雀魂の牌譜（四人打ち）を読む。復号した JSON のうち、head.result.players の各席の part_point_1（終局の点）、head.accounts の
 * 各席の nickname（打ち手の名前）、records の各要素 {name: ".lq.RecordNewRound" など, data: {...}} を読む。キーの名前は
 * snake_case でも lowerCamelCase でもよい。牌譜を取る API の応答の protobuf（ResGameRecord を包んだ lq.Wrapper）は、
 * {@link MajSoulProto} で同じ形の JSON に直してから読む。ルールは、局の始まりの scores の席の数で決める。三人打ちの牌譜は読まない。
 *
 * <p>牌の表記から TileCopies の割り当て方で物理 id を決め、イベントは RecordedRound の契約の順に並べ直す。親の配牌 14 枚の
 * 最後の 1 枚は最初のツモにする。立直の成立は、宣言牌の後の記録が持つ liqi（failed でないもの）で知る。表示牌は、doras の列で
 * 増えた分をめくる。
 *
 * <p>和了は hules の fans・count・fu から作る。役満は yiman で示し、count が倍率になる。放銃者は、最後に打牌か加槓をした席。
 * 包の席は、和了者の面子から決める（大三元なら 3 つ目の三元牌、大四喜なら 4 つ目の風牌の面子を鳴かせた席）。
 * 点の動きは delta_scores から作り（荒牌では scores の各要素が持つ delta_scores の和）、この局で成立した立直の供託を引く。
 * 流し満貫の席は、荒牌の精算が供託を引く前の点の動きと一致する席の組として決める。
 */
public final class MajSoulReader {

  /** records の name の前に付く名前空間。 */
  private static final String NAMESPACE = ".lq.";

  /** 記録の種類。name は "Record" の後に定数の名前を UpperCamelCase で続けたもの（NEW_ROUND なら RecordNewRound）。 */
  private enum Record {
    NEW_ROUND,
    DEAL_TILE,
    DISCARD_TILE,
    CHI_PENG_GANG,
    AN_GANG_ADD_GANG,
    HULE,
    NO_TILE,
    LIU_JU;

    static Record parse(String name) {
      String bare = name.startsWith(NAMESPACE) ? name.substring(NAMESPACE.length()) : name;
      for (Record record : values()) {
        if (("Record" + camel(record.name(), true)).equals(bare)) {
          return record;
        }
      }
      throw new IllegalArgumentException("Unknown Mahjong Soul record " + name);
    }
  }

  /** RecordChiPengGang の type。値は定数を書いた順に 0 から。 */
  private enum CallType {
    CHI,
    PON,
    DAIMINKAN
  }

  /** RecordAnGangAddGang の type。 */
  private enum KanType {
    KAKAN(2),
    ANKAN(3);

    private final int code;

    KanType(int code) {
      this.code = code;
    }

    static KanType of(int code) {
      for (KanType type : values()) {
        if (type.code == code) {
          return type;
        }
      }
      throw new IllegalArgumentException("Unknown Mahjong Soul kan type " + code);
    }
  }

  /** RecordLiuJu の type。値は定数を書いた順に 1 から。 */
  private enum Abort {
    KYUSHU_KYUHAI(AbortiveDraw.NINE_TERMINALS),
    SUFON_RENDA(AbortiveDraw.FOUR_WINDS),
    SUKAN_SANRA(AbortiveDraw.FOUR_KANS),
    SUCHA_RIICHI(AbortiveDraw.FOUR_RIICHI),
    SANCHAHO(AbortiveDraw.TRIPLE_RON);

    private final AbortiveDraw draw;

    Abort(AbortiveDraw draw) {
      this.draw = draw;
    }
  }

  /** hules の fans の id。値は定数を書いた順に 1 から。ドラ・赤・裏ドラ・抜いた北は役でなく、翻の数だけを持つ。 */
  private enum Fan {
    MENZEN_TSUMO,
    RIICHI,
    CHANKAN,
    RINSHAN,
    HAITEI,
    HOUTEI,
    YAKUHAI_HAKU,
    YAKUHAI_HATSU,
    YAKUHAI_CHUN,
    YAKUHAI_SEAT,
    YAKUHAI_ROUND,
    TANYAO,
    IPEIKOU,
    PINFU,
    CHANTA,
    ITTSU,
    SANSHOKU,
    DOUBLE_RIICHI,
    SANSHOKU_DOUKOU,
    SANKANTSU,
    TOITOI,
    SANANKOU,
    SHOUSANGEN,
    HONROUTOU,
    CHIITOITSU,
    JUNCHAN,
    HONITSU,
    RYANPEIKOU,
    CHINITSU,
    IPPATSU,
    DORA(null),
    AKA(null),
    URA(null),
    NUKI(null),
    TENHOU,
    CHIIHOU,
    DAISANGEN,
    SUUANKOU,
    TSUUIISOU,
    RYUUIISOU,
    CHINROUTOU,
    KOKUSHI,
    SHOUSUUSHII,
    SUUKANTSU,
    CHUUREN,
    /** 天鳳のルールに無い役。 */
    PAARENCHAN(null),
    JUNSEI_CHUUREN(ScoringYaku.CHUUREN),
    SUUANKOU_TANKI(ScoringYaku.SUUANKOU),
    KOKUSHI_13(ScoringYaku.KOKUSHI),
    DAISUUSHII;

    private final ScoringYaku scoring;

    Fan() {
      scoring = ScoringYaku.valueOf(name());
    }

    Fan(ScoringYaku scoring) {
      this.scoring = scoring;
    }

    static Fan of(int id) {
      if (id < 1 || id > values().length) {
        throw new IllegalArgumentException("Unknown Mahjong Soul fan " + id);
      }
      return values()[id - 1];
    }

    ScoringYaku scoring() {
      if (scoring == null) {
        throw new IllegalArgumentException("Mahjong Soul fan " + this + " is not a yaku of the Tenhou rules");
      }
      return scoring;
    }
  }

  private MajSoulReader() {}

  /**
   * 雀魂の牌譜の JSON を読む。読めない牌譜には IllegalArgumentException を投げる。
   *
   * @param json 復号した牌譜の JSON
   * @param id 牌譜の id
   * @return 読んだ半荘
   */
  public static RecordedGame read(String json, String id) {
    return read(JsonParser.parseString(json).getAsJsonObject(), id);
  }

  /**
   * 雀魂の牌譜を取る API の応答の protobuf（ResGameRecord を包んだ lq.Wrapper）を読む。読めない牌譜には IllegalArgumentException を
   * 投げる。
   *
   * @param response 応答のバイト列
   * @param id 牌譜の id
   * @return 読んだ半荘
   */
  public static RecordedGame read(byte[] response, String id) {
    return read(MajSoulProto.document(response), id);
  }

  /** バイト列が、雀魂の牌譜を取る API の応答の protobuf（ResGameRecord を包んだ lq.Wrapper）か。先頭の name で見分ける。 */
  public static boolean isResponse(byte[] bytes) {
    return MajSoulProto.isResponse(bytes);
  }

  private static RecordedGame read(JsonObject document, String id) {
    Reader reader = new Reader(id);
    for (JsonElement record : field(document, "records").getAsJsonArray()) {
      reader.accept(record.getAsJsonObject());
    }
    JsonObject head = field(document, "head").getAsJsonObject();
    JsonObject result = field(head, "result").getAsJsonObject();
    return reader.game(field(result, "players").getAsJsonArray(), array(head, "accounts"));
  }

  /** キーの値を返す。snake_case の名前で見つからなければ、lowerCamelCase の名前で探す。どちらも無いか、値が null なら null を返す。 */
  private static JsonElement find(JsonObject object, String key) {
    JsonElement value = object.get(key);
    if (value == null) {
      value = object.get(camel(key, false));
    }
    return value == null || value.isJsonNull() ? null : value;
  }

  private static JsonElement field(JsonObject object, String key) {
    JsonElement value = find(object, key);
    if (value == null) {
      throw new IllegalArgumentException("Mahjong Soul record has no " + key);
    }
    return value;
  }

  private static int integer(JsonObject object, String key) {
    return field(object, key).getAsInt();
  }

  private static boolean flag(JsonObject object, String key) {
    JsonElement value = find(object, key);
    return value != null && value.getAsBoolean();
  }

  private static Face face(JsonElement value) {
    return Face.parse(value.getAsString());
  }

  /** キーの値の配列を返す。キーが無ければ空の配列を返す。 */
  private static JsonArray array(JsonObject object, String key) {
    JsonElement value = find(object, key);
    return value == null ? new JsonArray() : value.getAsJsonArray();
  }

  private static int[] integers(JsonArray values) {
    int[] integers = new int[values.size()];
    for (int index = 0; index < integers.length; index++) {
      integers[index] = values.get(index).getAsInt();
    }
    return integers;
  }

  /**
   * snake_case（大文字でもよい）の名前を CamelCase に直す。
   *
   * @param upper true なら UpperCamelCase、false なら lowerCamelCase
   */
  static String camel(String snake, boolean upper) {
    StringBuilder camel = new StringBuilder();
    boolean start = upper;
    for (char c : snake.toCharArray()) {
      if (c == '_') {
        start = true;
      } else {
        camel.append(start ? Character.toUpperCase(c) : Character.toLowerCase(c));
        start = false;
      }
    }
    return camel.toString();
  }

  /** 1 つの牌譜を読む。局の状態は RecordNewRound ごとに始め直す。 */
  private static final class Reader {

    private final String id;
    private final List<RecordedRound> rounds = new ArrayList<>();
    private final List<RecordEvent> events = new ArrayList<>();
    private final List<RoundResult.Win> wins = new ArrayList<>();
    private Variant variant;
    private int seats;
    private TileCopies copies;

    private RoundStart start;
    private int[][] deal;
    private int indicator;

    /** めくった表示牌の数（doras の列のうち、読んだ所まで）。 */
    private int indicators;

    private int lastDiscarder;
    private int lastDiscard;

    /** ロンされうる牌を最後に出した席（打牌か加槓をした席）。 */
    private int claimant;

    /** この局で立直が成立した席（席ごとのビット）。 */
    private int accepted;

    private EndKind kind;
    private AbortiveDraw abortive;
    private boolean nagashi;
    private int tenpaiMask;
    private int[] moved;
    private int[] ura;

    Reader(String id) {
      this.id = Objects.requireNonNull(id, "id");
    }

    void accept(JsonObject wrapper) {
      Record record = Record.parse(field(wrapper, "name").getAsString());
      JsonObject data = field(wrapper, "data").getAsJsonObject();
      if (record == Record.NEW_ROUND) {
        finishRound();
        startRound(data);
        return;
      }
      if (start == null || kind != null) {
        throw new IllegalArgumentException(id + " has " + record + " outside a round");
      }
      switch (record) {
        case DEAL_TILE -> {
          riichi(data);
          reveal(data);
          int seat = seat(integer(data, "seat"));
          events.add(new RecordEvent.Draw(seat, copies.draw(seat, face(field(data, "tile")))));
        }
        case DISCARD_TILE -> discard(data);
        case CHI_PENG_GANG -> {
          riichi(data);
          call(data);
        }
        case AN_GANG_ADD_GANG -> kan(data);
        case HULE -> win(data);
        case NO_TILE -> exhaustive(data);
        case LIU_JU -> {
          riichi(data);
          kind = EndKind.ABORTIVE;
          int type = integer(data, "type");
          if (type < 1 || type > Abort.values().length) {
            throw new IllegalArgumentException(id + " has an unknown abortive draw " + type);
          }
          abortive = Abort.values()[type - 1].draw;
          moved = new int[seats];
        }
      }
    }

    /**
     * 最後の局を読み終え、終局の点と打ち手の名前を読んで、半荘を返す。
     *
     * @param players 各席の part_point_1（終局の点）を持つ配列
     * @param accounts 各席の nickname（打ち手の名前）を持つ配列。載っていない席と、名前の無い席は空の文字列にする
     */
    RecordedGame game(JsonArray players, JsonArray accounts) {
      finishRound();
      if (rounds.isEmpty()) {
        throw new IllegalArgumentException(id + " has no round");
      }
      if (players.size() != seats) {
        throw new IllegalArgumentException(id + " has final scores for " + players.size() + " of " + seats + " seats");
      }
      int[] finalScores = new int[seats];
      for (JsonElement player : players) {
        JsonObject object = player.getAsJsonObject();
        finalScores[seat(integer(object, "seat"))] = integer(object, "part_point_1");
      }
      String[] names = new String[seats];
      Arrays.fill(names, "");
      for (JsonElement account : accounts) {
        JsonObject object = account.getAsJsonObject();
        JsonElement nickname = find(object, "nickname");
        names[seat(integer(object, "seat"))] = nickname == null ? "" : nickname.getAsString();
      }
      RecordedGame.Meta meta = new RecordedGame.Meta(id, RecordedGame.Source.MAJSOUL, List.of(names));
      return new RecordedGame(variant, rounds, finalScores, meta);
    }

    private void startRound(JsonObject data) {
      int[] scores = integers(field(data, "scores").getAsJsonArray());
      if (variant == null) {
        variant =
            Arrays.stream(Variant.values())
                .filter(each -> each.seats() == scores.length)
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException(id + " has scores for " + scores.length + " seats"));
        if (variant.has(RuleFlag.NUKI)) {
          throw new IllegalArgumentException(id + " is a three-player game, which this reader does not read");
        }
        seats = variant.seats();
        copies = new TileCopies(variant);
      }
      int round = integer(data, "chang") * seats + integer(data, "ju");
      start = new RoundStart(variant, round, integer(data, "ben"), integer(data, "liqibang"), scores);
      int dealer = variant.dealer(round);
      copies.clear();
      deal = new int[seats][Variant.HAND];
      Face firstDraw = null;
      for (int seat = 0; seat < seats; seat++) {
        JsonArray tiles = field(data, "tiles" + seat).getAsJsonArray();
        if (tiles.size() != Variant.HAND + (seat == dealer ? 1 : 0)) {
          throw new IllegalArgumentException(id + " deals " + tiles.size() + " tiles to seat " + seat);
        }
        for (int index = 0; index < Variant.HAND; index++) {
          deal[seat][index] = copies.deal(seat, face(tiles.get(index)));
        }
        if (seat == dealer) {
          firstDraw = face(tiles.get(Variant.HAND));
        }
      }
      JsonArray doras = array(data, "doras");
      indicator = copies.reveal(doras.isEmpty() ? face(field(data, "dora")) : face(doras.get(0)));
      indicators = 1;
      events.clear();
      wins.clear();
      lastDiscarder = TurnEvent.NONE;
      lastDiscard = TurnEvent.NONE;
      claimant = TurnEvent.NONE;
      accepted = 0;
      kind = null;
      abortive = null;
      nagashi = false;
      tenpaiMask = 0;
      moved = null;
      ura = new int[0];
      events.add(new RecordEvent.Draw(dealer, copies.draw(dealer, firstDraw)));
    }

    private void discard(JsonObject data) {
      int seat = seat(integer(data, "seat"));
      int drawn = copies.drawn(seat);
      int tile = copies.discard(seat, face(field(data, "tile")), flag(data, "moqie"));
      boolean riichi = flag(data, "is_liqi") || flag(data, "is_wliqi");
      events.add(new RecordEvent.Discard(seat, tile, tile == drawn, riichi));
      lastDiscarder = seat;
      lastDiscard = tile;
      claimant = seat;
      reveal(data);
    }

    private void call(JsonObject data) {
      int seat = seat(integer(data, "seat"));
      JsonArray tiles = field(data, "tiles").getAsJsonArray();
      int[] froms = integers(field(data, "froms").getAsJsonArray());
      if (froms.length != tiles.size()) {
        throw new IllegalArgumentException(
            id + " has a call of " + tiles.size() + " tiles from " + froms.length + " seats");
      }
      Face called = null;
      int target = TurnEvent.NONE;
      List<Face> consumed = new ArrayList<>();
      for (int index = 0; index < froms.length; index++) {
        if (froms[index] == seat) {
          consumed.add(face(tiles.get(index)));
        } else if (called == null) {
          called = face(tiles.get(index));
          target = froms[index];
        } else {
          throw new IllegalArgumentException(id + " has a call of two tiles from other seats");
        }
      }
      if (called == null
          || target != lastDiscarder
          || called.kind() != Tiles.kind(lastDiscard)
          || called.red() != variant.tiles().red(lastDiscard)) {
        throw new IllegalArgumentException(id + ": seat " + seat + " calls a tile that is not the last discard");
      }
      Meld.Type type = Meld.Type.valueOf(callType(integer(data, "type")).name());
      Meld meld = Meld.call(type, Meld.RelativeSource.of(seat, target, seats), called, consumed);
      for (Face face : consumed) {
        copies.meld(seat, face);
      }
      events.add(new RecordEvent.Call(seat, meld.id()));
    }

    private void kan(JsonObject data) {
      int seat = seat(integer(data, "seat"));
      Face tile = face(field(data, "tiles"));
      switch (KanType.of(integer(data, "type"))) {
        case ANKAN -> {
          for (int copy = 0; copy < Tiles.COPIES; copy++) {
            copies.meld(seat, Face.of(tile.kind(), variant.tiles().red(Tiles.physicalId(tile.kind(), copy))));
          }
          events.add(new RecordEvent.ClosedKan(seat, tile.kind()));
        }
        case KAKAN -> {
          copies.meld(seat, tile);
          events.add(new RecordEvent.AddedKan(seat, tile.kind()));
          claimant = seat;
        }
      }
      reveal(data);
    }

    private void win(JsonObject data) {
      JsonArray hules = field(data, "hules").getAsJsonArray();
      boolean tsumo = !hules.isEmpty() && flag(hules.get(0).getAsJsonObject(), "zimo");
      if (hules.isEmpty() || tsumo && hules.size() > 1) {
        throw new IllegalArgumentException(id + " has a RecordHule of " + hules.size() + " wins");
      }
      for (JsonElement value : hules) {
        JsonObject hule = value.getAsJsonObject();
        JsonArray uraIndicators = array(hule, "li_doras");
        if (ura.length == 0 && !uraIndicators.isEmpty()) {
          ura = new int[uraIndicators.size()];
          for (int index = 0; index < ura.length; index++) {
            ura[index] = copies.reveal(face(uraIndicators.get(index)));
          }
        }
        wins.add(scored(hule, seat(integer(hule, "seat"))));
      }
      kind = tsumo ? EndKind.TSUMO : EndKind.RON;
      if (!tsumo) {
        wins.sort(Comparator.comparingInt(win -> Math.floorMod(win.seat() - claimant, seats)));
      }
      moved = integers(field(data, "delta_scores").getAsJsonArray());
    }

    /** hule の役・翻・符から和了を作る。役満は yiman で示し、count が倍率になる。 */
    private RoundResult.Win scored(JsonObject hule, int seat) {
      long yaku = 0;
      int dora = 0;
      int aka = 0;
      int uraHan = 0;
      for (JsonElement value : array(hule, "fans")) {
        JsonObject fan = value.getAsJsonObject();
        Fan each = Fan.of(integer(fan, "id"));
        int han = integer(fan, "val");
        switch (each) {
          case DORA, NUKI -> dora += han;
          case AKA -> aka += han;
          case URA -> uraHan += han;
          default -> yaku |= han > 0 ? each.scoring().bit() : 0;
        }
      }
      int count = integer(hule, "count");
      int fu = find(hule, "fu") == null ? 0 : integer(hule, "fu");
      int loser = flag(hule, "zimo") ? RoundOutcome.NO_SEAT : claimant;
      return RoundResult.Win.of(
          seat, yaku, count, fu, dora, aka, uraHan, flag(hule, "yiman") ? count : 0, pao(seat, yaku, loser));
    }

    /**
     * 和了の包の席を返す。和了者の面子を足した順に数え、大三元なら 3 つ目の三元牌の面子、大四喜なら 4 つ目の風牌の面子を鳴かせた
     * 席が包になる（加槓した面子は、元のポンの所で数える）。その面子が暗槓のとき、鳴かせた席が放銃者のとき、どちらの役も無いときは
     * {@code RoundOutcome.NO_SEAT} を返す。
     *
     * @param seat 和了者の席
     * @param yaku 和了の役のビットの論理和
     * @param loser ロンの放銃者。ツモなら {@code RoundOutcome.NO_SEAT}
     */
    private int pao(int seat, long yaku, int loser) {
      Tiles.Group group;
      if ((yaku & ScoringYaku.DAISANGEN.bit()) != 0) {
        group = Tiles.Group.DRAGON;
      } else if ((yaku & ScoringYaku.DAISUUSHII.bit()) != 0) {
        group = Tiles.Group.WIND;
      } else {
        return RoundOutcome.NO_SEAT;
      }
      int sets = 0;
      for (RecordEvent event : events) {
        int kind;
        int feeder;
        if (event instanceof RecordEvent.Call call && call.seat() == seat) {
          Meld meld = Meld.of(call.meld());
          kind = meld.base();
          feeder = meld.source().seat(seat, seats);
        } else if (event instanceof RecordEvent.ClosedKan kan && kan.seat() == seat) {
          kind = kan.kind();
          feeder = RoundOutcome.NO_SEAT;
        } else {
          continue;
        }
        if (Tiles.Group.of(kind) == group && ++sets == group.size()) {
          return feeder == loser ? RoundOutcome.NO_SEAT : feeder;
        }
      }
      return RoundOutcome.NO_SEAT;
    }

    private void exhaustive(JsonObject data) {
      kind = EndKind.EXHAUSTIVE;
      nagashi = flag(data, "liujumanguan");
      JsonArray players = array(data, "players");
      for (int seat = 0; seat < players.size(); seat++) {
        tenpaiMask |= flag(players.get(seat).getAsJsonObject(), "tingpai") ? 1 << seat : 0;
      }
      moved = new int[seats];
      for (JsonElement score : array(data, "scores")) {
        int[] deltas = integers(array(score.getAsJsonObject(), "delta_scores"));
        for (int seat = 0; seat < deltas.length; seat++) {
          moved[seat] += deltas[seat];
        }
      }
    }

    /** liqi（failed でないもの）があれば、その席の立直を成立させる。 */
    private void riichi(JsonObject data) {
      JsonElement liqi = find(data, "liqi");
      if (liqi == null || flag(liqi.getAsJsonObject(), "failed")) {
        return;
      }
      int seat = seat(integer(liqi.getAsJsonObject(), "seat"));
      accepted |= 1 << seat;
      events.add(new RecordEvent.RiichiAccepted(seat));
    }

    /** doras の列のうち、まだめくっていない表示牌をめくる。 */
    private void reveal(JsonObject data) {
      JsonArray doras = array(data, "doras");
      for (; indicators < doras.size(); indicators++) {
        events.add(new RecordEvent.DoraReveal(copies.reveal(face(doras.get(indicators)))));
      }
    }

    private void finishRound() {
      if (start == null) {
        return;
      }
      if (kind == null) {
        throw new IllegalArgumentException(id + " has no result in round " + start.round());
      }
      if (moved.length != seats) {
        throw new IllegalArgumentException(id + " moves points of " + moved.length + " seats");
      }
      int[] deltas = moved.clone();
      for (int seat = 0; seat < seats; seat++) {
        if ((accepted & 1 << seat) != 0) {
          deltas[seat] -= Variant.DEPOSIT;
        }
      }
      int loser = kind == EndKind.RON ? claimant : RoundOutcome.NO_SEAT;
      RoundResult result =
          new RoundResult(kind, abortive, wins, loser, tenpaiMask, nagashi ? nagashiMask() : 0, deltas, ura);
      rounds.add(new RecordedRound(start, deal, indicator, RecordedRound.contractOrder(events), result));
      start = null;
    }

    /**
     * 流し満貫の席を、席ごとのビットで返す。荒牌の精算が点の動き（立直の供託を引く前）と一致する、空でない席の組を探す。見つからなければ
     * IllegalArgumentException を投げる。
     */
    private int nagashiMask() {
      int[] paid = new int[seats];
      for (int mask = 1; mask <= variant.seatMask(); mask++) {
        Arrays.fill(paid, 0);
        Settlement.exhaustive(variant, variant.dealer(start.round()), tenpaiMask, mask, paid);
        if (Arrays.equals(paid, moved)) {
          return mask;
        }
      }
      throw new IllegalArgumentException(id + ": no nagashi mangan pays " + Arrays.toString(moved));
    }

    private CallType callType(int code) {
      if (code < 0 || code >= CallType.values().length) {
        throw new IllegalArgumentException(id + " has an unknown call type " + code);
      }
      return CallType.values()[code];
    }

    private int seat(int seat) {
      if (seat < 0 || seat >= seats) {
        throw new IllegalArgumentException(id + " names seat " + seat + " of " + seats);
      }
      return seat;
    }
  }
}
