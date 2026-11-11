package com.epsilon.mahjong.io.mjai;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.record.TileCopies;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * mjai の牌譜を読み書きする（四人打ちと三人打ち）。読める形は 2 つあり、1 行に 1 つずつ JSON のイベントを並べたものと、イベントを
 * JSON の配列にしたもの。ルールは人数で決まり、四人打ちは {@link Variant#TENHOU_4P}、三人打ちは {@link Variant#TENHOU_3P} になる。
 * 人数は start_game の names の数で数える。start_game に names が無ければ（RiichiEnv の書き方）、最初の start_kyoku の scores の数で
 * 数え、打ち手の名前は全て空の文字列にする。names があるときは、scores の数が names と同じでなければならない。null の名前は空の
 * 文字列にする。none のイベントは読み飛ばす。
 *
 * <p>三人打ちの牌譜では、席ごとの値（names・scores・tehais・deltas・tenpais）が 3 つずつで、局は 1〜3。北抜きは RiichiEnv と同じ
 * {@code {"type":"kita","actor":席,"pai":"N"}} で、type は kita だけを読む。pai は省いてよく（省けば北）、書くなら北でなければ
 * ならない。抜いた北にはロンできる。北抜きより前の大明槓・加槓の後めくりの dora は、RiichiEnv と同じく北抜きと補充のツモの間に書く。
 * 三人打ちにチーは無く、北のポン・槓はふつうの面子として扱う。hora の yakus の nukidora は抜いた北の翻で、読むときはドラの翻に足す
 * （役の名前は nukidora だけ）。四人打ちの牌譜に北抜きがあるか、三人打ちの牌譜にチーがあれば IllegalArgumentException を投げる。
 * 大明槓の type は kan も読む。
 *
 * <p>読むときは、牌の表記から TileCopies の割り当て方で物理 id を決め、イベントを RecordedRound の契約どおりの順に並べ直す。和了は
 * hora の yakus・fu・fan（役満は役 1 つにつき 100）から作る。hora_points が翻・符から計算した和了点と違えば
 * IllegalArgumentException を投げる。sangenpai の翻は三元牌の刻子の数でしかないので、どの三元牌かは、和了の形（手牌・面子・ロンの
 * 牌）に 3 枚以上ある三元牌で決める。yakus・fu・fan・hora_points のどれも無い hora は、点の内訳を持たない和了にする。
 * 一部だけを持つ hora は IllegalArgumentException を投げる。ダブロンの和了は、放銃者から手番の近い順に並べる。
 *
 * <p>裏ドラ表示牌は、uradora_markers か ura_markers のうち空でない最初のものを使う（立直していない和了者に空の配列を書く記録がある）。
 * 局の点の移動は、hora・ryukyoku の deltas を足し、この局で成立した立直の供託を引いて求める。包の席は hora の pao から読み、pao が
 * 無ければ包は無い。ryukyoku の終わり方は reason で決める（天鳳の名前などの別名も読む）。荒牌なら tenpais で聴牌の席を決め、荒牌の
 * 精算が deltas と一致する席の組を流し満貫の席にする。reason が無いか、荒牌なのに tenpais が無い ryukyoku は、終わり方を持たない
 * 流局にする。end_game に scores が無ければ、最後の局の始まりの点に点の移動を足し、場に残った供託をトップに渡して終局の点を求める。
 *
 * <p>書くときも同じ形で書き、記録に無い内訳は書かない。大明槓・加槓の後めくりの dora は、mjai の並びのとおり打牌の前に書く。
 * 三人打ちの和了では、ドラの翻のうち、和了者がその局で抜いた北の数を nukidora に分けて書く。
 */
public final class MjaiCodec {

  /** 役満の役 1 つあたりの fan。 */
  private static final int YAKUMAN_FAN = 100;

  private static final int YAKUMAN_BASE = BaseClass.YAKUMAN.base();

  /** 三元牌の役。三元牌の種類の順と同じく、白・發・中の順に並ぶ。 */
  private static final ScoringYaku[] DRAGONS = {
    ScoringYaku.YAKUHAI_HAKU, ScoringYaku.YAKUHAI_HATSU, ScoringYaku.YAKUHAI_CHUN
  };

  private static final String LINE = "\n";

  /** JSON の配列で書いた牌譜の先頭の文字。 */
  private static final String ARRAY = "[";

  /** 和了の内訳のキー。hora は全てを持つか、どれも持たない。 */
  private static final List<String> BREAKDOWN = List.of("yakus", "fan", "fu", "hora_points");

  /**
   * イベントの type。文字列は定数の名前の小文字で、読むときは別名も受け付ける。定数ごとに、局の中でだけ起きるイベントかを持つ。
   * none のイベントはどこにあってもよく、読み飛ばす。
   */
  private enum Type {
    START_GAME(false),
    START_KYOKU(false),
    TSUMO(true),
    DAHAI(true),
    CHI(true),
    PON(true),
    DAIMINKAN(true, "kan"),
    ANKAN(true),
    KAKAN(true),
    KITA(true),
    DORA(true),
    REACH(true),
    REACH_ACCEPTED(true),
    HORA(true),
    RYUKYOKU(true),
    END_KYOKU(true),
    END_GAME(false),
    NONE(false);

    private final boolean inRound;
    private final List<String> aliases;

    Type(boolean inRound, String... aliases) {
      this.inRound = inRound;
      this.aliases = List.of(aliases);
    }

    String text() {
      return name().toLowerCase(Locale.ROOT);
    }

    static Type parse(String text) {
      for (Type type : values()) {
        if (type.text().equals(text) || type.aliases.contains(text)) {
          return type;
        }
      }
      throw new IllegalArgumentException("Unknown mjai event type " + text);
    }
  }

  /** ryukyoku の reason。文字列は定数の名前の小文字で、読むときは別名も受け付ける。fanpai と nagashimangan は荒牌。 */
  private enum Reason {
    FANPAI(null, "exhaustive_draw"),
    NAGASHIMANGAN(null),
    KYUSHUKYUHAI(AbortiveDraw.NINE_TERMINALS, "kyushu_kyuhai", "yao9"),
    SUFONRENTA(AbortiveDraw.FOUR_WINDS, "kaze4"),
    SUCHAREACH(AbortiveDraw.FOUR_RIICHI, "reach4"),
    SUKAIKAN(AbortiveDraw.FOUR_KANS, "kan4"),
    SANCHAHO(AbortiveDraw.TRIPLE_RON, "ron3");

    private final AbortiveDraw abortive;
    private final List<String> aliases;

    Reason(AbortiveDraw abortive, String... aliases) {
      this.abortive = abortive;
      this.aliases = List.of(aliases);
    }

    String text() {
      return name().toLowerCase(Locale.ROOT);
    }

    static Reason parse(String text) {
      for (Reason reason : values()) {
        if (reason.text().equals(text) || reason.aliases.contains(text)) {
          return reason;
        }
      }
      throw new IllegalArgumentException("Unknown mjai ryukyoku reason " + text);
    }

    static Reason of(RoundResult result) {
      if (result.kind() == EndKind.EXHAUSTIVE) {
        return result.nagashiMask() != 0 ? NAGASHIMANGAN : FANPAI;
      }
      for (Reason reason : values()) {
        if (reason.abortive == result.abortive()) {
          return reason;
        }
      }
      throw new IllegalArgumentException("mjai has no ryukyoku reason for " + result.abortive());
    }
  }

  /**
   * hora の yakus に書く役の名前（定数の名前の小文字）。dora・uradora・akadora・nukidora は翻の数だけを持ち、sangenpai は
   * 三元牌の刻子の数を持つ。
   */
  private enum Yaku {
    MENZENCHIN_TSUMOHO(ScoringYaku.MENZEN_TSUMO),
    REACH(ScoringYaku.RIICHI),
    IPPATSU(ScoringYaku.IPPATSU),
    CHANKAN(ScoringYaku.CHANKAN),
    RINSHANKAIHO(ScoringYaku.RINSHAN),
    HAITEIRAOYUE(ScoringYaku.HAITEI),
    HOTEIRAOYUI(ScoringYaku.HOUTEI),
    PINFU(ScoringYaku.PINFU),
    TANYAOCHU(ScoringYaku.TANYAO),
    IPEKO(ScoringYaku.IPEIKOU),
    JIKAZE(ScoringYaku.YAKUHAI_SEAT),
    BAKAZE(ScoringYaku.YAKUHAI_ROUND),
    SANGENPAI(null),
    DOUBLE_REACH(ScoringYaku.DOUBLE_RIICHI),
    CHITOITSU(ScoringYaku.CHIITOITSU),
    HONCHANTAIYAO(ScoringYaku.CHANTA),
    IKKITSUKAN(ScoringYaku.ITTSU),
    SANSHOKUDOJUN(ScoringYaku.SANSHOKU),
    SANSHOKUDOKO(ScoringYaku.SANSHOKU_DOUKOU),
    SANKANTSU(ScoringYaku.SANKANTSU),
    TOITOIHO(ScoringYaku.TOITOI),
    SANANKO(ScoringYaku.SANANKOU),
    SHOSANGEN(ScoringYaku.SHOUSANGEN),
    HONROTO(ScoringYaku.HONROUTOU),
    RYANPEKO(ScoringYaku.RYANPEIKOU),
    JUNCHANTAIYAO(ScoringYaku.JUNCHAN),
    HONISO(ScoringYaku.HONITSU),
    CHINISO(ScoringYaku.CHINITSU),
    TENHO(ScoringYaku.TENHOU),
    CHIHO(ScoringYaku.CHIIHOU),
    DAISANGEN(ScoringYaku.DAISANGEN),
    SUANKO(ScoringYaku.SUUANKOU),
    TSUISO(ScoringYaku.TSUUIISOU),
    RYUISO(ScoringYaku.RYUUIISOU),
    CHINROTO(ScoringYaku.CHINROUTOU),
    CHURENPOTON(ScoringYaku.CHUUREN),
    KOKUSHIMUSO(ScoringYaku.KOKUSHI),
    DAISUSHI(ScoringYaku.DAISUUSHII),
    SHOSUSHI(ScoringYaku.SHOUSUUSHII),
    SUKANTSU(ScoringYaku.SUUKANTSU),
    DORA(null),
    URADORA(null),
    AKADORA(null),
    NUKIDORA(null);

    private final ScoringYaku scoring;

    Yaku(ScoringYaku scoring) {
      this.scoring = scoring;
    }

    String text() {
      return name().toLowerCase(Locale.ROOT);
    }

    static Yaku parse(String text) {
      for (Yaku yaku : values()) {
        if (yaku.text().equals(text)) {
          return yaku;
        }
      }
      throw new IllegalArgumentException("Unknown mjai yaku " + text);
    }

    /** 点数計算の役を、mjai に書く役に直す。三元牌の役は sangenpai にまとめ、ほかは同じ名前の役にする。 */
    static Yaku of(ScoringYaku scoring) {
      if (Arrays.asList(DRAGONS).contains(scoring)) {
        return SANGENPAI;
      }
      for (Yaku yaku : values()) {
        if (yaku.scoring == scoring) {
          return yaku;
        }
      }
      throw new IllegalArgumentException("mjai has no yaku for " + scoring);
    }
  }

  private MjaiCodec() {}

  /**
   * mjai の牌譜（1 行に 1 つずつ並べたイベントか、イベントの配列）を読む。読めない牌譜なら IllegalArgumentException を投げる。
   *
   * @param text 牌譜の文字列
   * @param id 牌譜 id
   * @return 読んだ半荘
   */
  public static RecordedGame read(String text, String id) {
    Reader reader = new Reader(id);
    if (text.strip().startsWith(ARRAY)) {
      for (JsonElement event : JsonParser.parseString(text).getAsJsonArray()) {
        reader.accept(event.getAsJsonObject());
      }
    } else {
      for (String line : text.split(LINE)) {
        if (!line.isBlank()) {
          reader.accept(JsonParser.parseString(line).getAsJsonObject());
        }
      }
    }
    return reader.game();
  }

  /** 半荘を mjai の牌譜に書く（1 行に 1 つのイベント）。 */
  public static String write(RecordedGame game) {
    return new Writer(game.variant()).write(game);
  }

  private static JsonElement field(JsonObject event, String key) {
    JsonElement value = optional(event, key);
    if (value == null) {
      throw new IllegalArgumentException("mjai " + event.get("type") + " has no " + key);
    }
    return value;
  }

  /**
   * イベントから、そのキーの値を読む。
   *
   * @param key キーの名前
   * @return キーの値。キーが無いか、値が null なら null
   */
  private static JsonElement optional(JsonObject event, String key) {
    JsonElement value = event.get(key);
    return value == null || value.isJsonNull() ? null : value;
  }

  private static int integer(JsonObject event, String key) {
    return field(event, key).getAsInt();
  }

  private static Face face(JsonObject event, String key) {
    return Face.parse(field(event, key).getAsString());
  }

  private static int[] integers(JsonObject event, String key, int size) {
    JsonArray values = field(event, key).getAsJsonArray();
    if (values.size() != size) {
      throw new IllegalArgumentException("mjai " + key + " needs " + size + " values, got " + values.size());
    }
    int[] integers = new int[size];
    for (int index = 0; index < size; index++) {
      integers[index] = values.get(index).getAsInt();
    }
    return integers;
  }

  private static List<Face> faces(JsonObject event, String key) {
    List<Face> faces = new ArrayList<>();
    for (JsonElement value : field(event, key).getAsJsonArray()) {
      faces.add(Face.parse(value.getAsString()));
    }
    return faces;
  }

  /** 1 つの牌譜を読む。局の状態は start_kyoku ごとに作り直す。 */
  private static final class Reader {

    private final String id;
    private final List<RecordedRound> rounds = new ArrayList<>();
    private final List<RecordEvent> events = new ArrayList<>();
    private final List<RoundResult.Win> wins = new ArrayList<>();
    private final List<String> names = new ArrayList<>();

    /** start_game を読んだか。 */
    private boolean started;

    /** ルール。names の無い start_game の後は、最初の start_kyoku まで null。 */
    private Variant variant;
    private TileSet set;
    private int seats;
    private TileCopies copies;
    private int[] finalScores;

    private RoundStart start;
    private int[][] deal;
    private int indicator;

    /** 席ごと・牌の種類ごとに、面子にある牌の数。 */
    private int[][] melded;

    private int lastDiscarder;
    private int lastDiscard;

    /** 次の打牌が立直の宣言牌になる席（席ごとのビット）。 */
    private int declared;

    /** この局で立直が成立した席（席ごとのビット）。 */
    private int accepted;

    /** いまの局を hora か ryukyoku で終えたか。 */
    private boolean ended;

    /** 局の終わり方。終わり方を持たない流局は null のまま。 */
    private EndKind kind;

    private AbortiveDraw abortive;
    private boolean nagashi;
    private int loser;
    private int tenpaiMask;

    /** 席ごとに足し上げた、hora・ryukyoku の deltas。 */
    private int[] moved;

    /** 裏ドラ表示牌の物理 id。読むまで null。 */
    private int[] ura;

    Reader(String id) {
      this.id = Objects.requireNonNull(id, "id");
    }

    void accept(JsonObject event) {
      Type type = Type.parse(field(event, "type").getAsString());
      if (type == Type.NONE) {
        return;
      }
      if ((type == Type.START_GAME) == started) {
        throw new IllegalArgumentException(id + " needs exactly one start_game first, got " + type.text());
      }
      if (type.inRound != (start != null)) {
        throw new IllegalArgumentException(
            id + " has " + type.text() + (type.inRound ? " outside" : " inside") + " a round");
      }
      switch (type) {
        case START_GAME -> startGame(event);
        case START_KYOKU -> startRound(event);
        case TSUMO -> {
          int actor = seat(integer(event, "actor"));
          events.add(new RecordEvent.Draw(actor, copies.draw(actor, face(event, "pai"))));
        }
        case DAHAI -> discard(event);
        case CHI, PON, DAIMINKAN -> call(event, Meld.Type.valueOf(type.name()));
        case ANKAN -> closedKan(event);
        case KAKAN -> {
          int actor = seat(integer(event, "actor"));
          Face added = face(event, "pai");
          copies.meld(actor, added);
          melded[actor][added.kind()]++;
          events.add(new RecordEvent.AddedKan(actor, added.kind()));
        }
        case KITA -> nuki(event);
        case DORA -> events.add(new RecordEvent.DoraReveal(copies.reveal(face(event, "dora_marker"))));
        case REACH -> declared |= 1 << seat(integer(event, "actor"));
        case REACH_ACCEPTED -> {
          int actor = seat(integer(event, "actor"));
          accepted |= 1 << actor;
          events.add(new RecordEvent.RiichiAccepted(actor));
        }
        case HORA -> win(event);
        case RYUKYOKU -> draw(event);
        case END_KYOKU -> finishRound();
        case END_GAME ->
            finalScores = optional(event, "scores") == null ? settledScores() : integers(event, "scores", seats);
        case NONE -> throw new IllegalStateException("A none event is skipped before the checks");
      }
    }

    RecordedGame game() {
      if (start != null || rounds.isEmpty() || finalScores == null) {
        throw new IllegalArgumentException(id + " ends before the game does");
      }
      RecordedGame.Meta meta = new RecordedGame.Meta(id, RecordedGame.Source.MJAI, names);
      return new RecordedGame(variant, rounds, finalScores, meta);
    }

    /** names があれば、その数でルールを決め、名前を席の順に覚える（null の名前は空の文字列）。無ければルールは最初の局で決める。 */
    private void startGame(JsonObject event) {
      started = true;
      JsonElement given = optional(event, "names");
      if (given == null) {
        return;
      }
      for (JsonElement name : given.getAsJsonArray()) {
        names.add(name.isJsonNull() ? "" : name.getAsString());
      }
      players(names.size());
    }

    /** その人数で打つルールに決める。 */
    private void players(int players) {
      variant =
          Arrays.stream(Variant.values())
              .filter(each -> each.seats() == players)
              .findFirst()
              .orElseThrow(() -> new IllegalArgumentException(id + " has " + players + " players"));
      set = variant.tiles();
      seats = variant.seats();
      copies = new TileCopies(variant);
      melded = new int[seats][Tiles.KINDS];
    }

    /** 局を始める。ルールがまだ無ければ、scores の数を人数にして、打ち手の名前を全て空の文字列にする。 */
    private void startRound(JsonObject event) {
      if (variant == null) {
        players(field(event, "scores").getAsJsonArray().size());
        names.addAll(Collections.nCopies(seats, ""));
      }
      int wind = face(event, "bakaze").kind() - Tiles.TON;
      int number = integer(event, "kyoku") - 1;
      if (wind < 0 || wind >= Tiles.Group.WIND.size() || number < 0 || number >= seats) {
        throw new IllegalArgumentException(id + " has no round " + event.get("bakaze") + event.get("kyoku"));
      }
      int round = wind * seats + number;
      int[] scores = integers(event, "scores", seats);
      start = new RoundStart(variant, round, integer(event, "honba"), integer(event, "kyotaku"), scores);
      if (integer(event, "oya") != variant.dealer(round)) {
        throw new IllegalArgumentException(id + " names dealer " + event.get("oya") + " in round " + round);
      }
      copies.clear();
      JsonArray hands = field(event, "tehais").getAsJsonArray();
      if (hands.size() != seats) {
        throw new IllegalArgumentException(id + " deals " + hands.size() + " hands to " + seats + " seats");
      }
      deal = new int[seats][];
      for (int seat = 0; seat < seats; seat++) {
        JsonArray hand = hands.get(seat).getAsJsonArray();
        deal[seat] = new int[hand.size()];
        for (int index = 0; index < hand.size(); index++) {
          deal[seat][index] = copies.deal(seat, Face.parse(hand.get(index).getAsString()));
        }
        Arrays.fill(melded[seat], 0);
      }
      indicator = copies.reveal(face(event, "dora_marker"));
      events.clear();
      wins.clear();
      lastDiscarder = TurnEvent.NONE;
      lastDiscard = TurnEvent.NONE;
      declared = 0;
      accepted = 0;
      ended = false;
      kind = null;
      abortive = null;
      nagashi = false;
      loser = RoundOutcome.NO_SEAT;
      tenpaiMask = 0;
      moved = new int[seats];
      ura = null;
    }

    private void discard(JsonObject event) {
      int actor = seat(integer(event, "actor"));
      int drawn = copies.drawn(actor);
      int tile = copies.discard(actor, face(event, "pai"), field(event, "tsumogiri").getAsBoolean());
      events.add(new RecordEvent.Discard(actor, tile, tile == drawn, (declared & 1 << actor) != 0));
      declared &= ~(1 << actor);
      lastDiscarder = actor;
      lastDiscard = tile;
    }

    private void call(JsonObject event, Meld.Type type) {
      int actor = seat(integer(event, "actor"));
      if (type == Meld.Type.CHI && !variant.has(RuleFlag.CHI)) {
        throw new IllegalArgumentException(id + ": seat " + actor + " calls chi in " + variant);
      }
      int target = seat(integer(event, "target"));
      Face called = face(event, "pai");
      if (target != lastDiscarder || called.kind() != Tiles.kind(lastDiscard) || called.red() != set.red(lastDiscard)) {
        throw new IllegalArgumentException(
            id + ": seat " + actor + " calls " + called.mjai() + " of seat " + target + ", not the last discard");
      }
      List<Face> consumed = faces(event, "consumed");
      Meld meld = Meld.call(type, Meld.RelativeSource.of(actor, target, seats), called, consumed);
      for (Face face : consumed) {
        copies.meld(actor, face);
      }
      for (int index = 0; index < meld.size(); index++) {
        melded[actor][meld.tile(index)]++;
      }
      events.add(new RecordEvent.Call(actor, meld.id()));
    }

    /** 北抜きを読む。pai が無ければ北を抜いたものとする。 */
    private void nuki(JsonObject event) {
      int actor = seat(integer(event, "actor"));
      if (!variant.has(RuleFlag.NUKI)) {
        throw new IllegalArgumentException(id + ": seat " + actor + " takes out a north in " + variant);
      }
      Face north = optional(event, "pai") == null ? Face.of(Tiles.PEI, false) : face(event, "pai");
      if (north.kind() != Tiles.PEI) {
        throw new IllegalArgumentException(id + ": seat " + actor + " takes out " + north.mjai() + ", not a north");
      }
      events.add(new RecordEvent.Nuki(actor, copies.meld(actor, north)));
    }

    private void closedKan(JsonObject event) {
      int actor = seat(integer(event, "actor"));
      List<Face> consumed = faces(event, "consumed");
      int kan = consumed.isEmpty() ? TurnEvent.NONE : consumed.get(0).kind();
      if (consumed.size() != Tiles.COPIES || consumed.stream().anyMatch(face -> face.kind() != kan)) {
        throw new IllegalArgumentException(id + ": seat " + actor + " cannot make an ankan of " + consumed);
      }
      for (Face face : consumed) {
        copies.meld(actor, face);
      }
      melded[actor][kan] += Tiles.COPIES;
      events.add(new RecordEvent.ClosedKan(actor, kan));
    }

    private void win(JsonObject event) {
      int actor = seat(integer(event, "actor"));
      int target = seat(integer(event, "target"));
      boolean tsumo = actor == target;
      if (ended && (tsumo || kind != EndKind.RON || target != loser)) {
        throw new IllegalArgumentException(id + " ends round " + start.round() + " twice");
      }
      readUra(event);
      wins.add(hasBreakdown(event) ? scoredWin(event, actor, tsumo) : RoundResult.Win.unscored(actor));
      ended = true;
      kind = tsumo ? EndKind.TSUMO : EndKind.RON;
      loser = tsumo ? RoundOutcome.NO_SEAT : target;
      add(integers(event, "deltas", seats));
    }

    /**
     * 裏ドラ表示牌を読む。uradora_markers か ura_markers が空でない、最初の hora の表示牌を使う。ダブロンで、立直していない和了者の
     * hora に空の配列があっても、後の和了者の表示牌を読む。
     */
    private void readUra(JsonObject event) {
      if (ura != null) {
        return;
      }
      String key = optional(event, "uradora_markers") != null ? "uradora_markers" : "ura_markers";
      if (optional(event, key) == null) {
        return;
      }
      List<Face> markers = faces(event, key);
      if (markers.isEmpty()) {
        return;
      }
      ura = new int[markers.size()];
      for (int index = 0; index < ura.length; index++) {
        ura[index] = copies.reveal(markers.get(index));
      }
    }

    /** hora が内訳のキーを全て持つか。どれも持たなければ偽を返し、一部だけなら IllegalArgumentException を投げる。 */
    private boolean hasBreakdown(JsonObject event) {
      long present = BREAKDOWN.stream().filter(key -> optional(event, key) != null).count();
      if (present != 0 && present != BREAKDOWN.size()) {
        throw new IllegalArgumentException(id + ": a hora has only part of " + BREAKDOWN);
      }
      return present != 0;
    }

    /** 内訳を持つ hora から和了を作る。hora_points が翻・符から計算した和了点と違えば IllegalArgumentException を投げる。 */
    private RoundResult.Win scoredWin(JsonObject event, int actor, boolean tsumo) {
      RoundResult.Win win = scored(event, actor, tsumo ? null : face(event, "pai"));
      int points = win.points(variant, variant.dealer(start.round()), tsumo);
      int recorded = integer(event, "hora_points");
      if (points != recorded) {
        throw new IllegalArgumentException(
            id + ": the hora of seat " + actor + " pays " + recorded + ", but " + win + " pays " + points);
      }
      return win;
    }

    /**
     * hora の役・翻・符から和了を作る。
     *
     * @param ron ロンの牌。ツモなら null
     */
    private RoundResult.Win scored(JsonObject event, int actor, Face ron) {
      long yaku = 0;
      int fan = integer(event, "fan");
      int dora = 0;
      int aka = 0;
      int uraHan = 0;
      for (JsonElement value : field(event, "yakus").getAsJsonArray()) {
        JsonArray pair = value.getAsJsonArray();
        Yaku each = Yaku.parse(pair.get(0).getAsString());
        int han = pair.get(1).getAsInt();
        switch (each) {
          case DORA, NUKIDORA -> dora += han;
          case URADORA -> uraHan += han;
          case AKADORA -> aka += han;
          case SANGENPAI -> yaku |= dragons(actor, ron, han);
          default -> yaku |= each.scoring.bit();
        }
      }
      int pao = optional(event, "pao") != null ? seat(integer(event, "pao")) : RoundOutcome.NO_SEAT;
      return RoundResult.Win.of(actor, yaku, fan, integer(event, "fu"), dora, aka, uraHan, fan / YAKUMAN_FAN, pao);
    }

    /**
     * 和了の形（手牌・面子・ロンの牌）に 3 枚以上ある三元牌の役を返す。
     *
     * @param count sangenpai の翻（三元牌の刻子の数）。見つけた数と違えば IllegalArgumentException を投げる
     */
    private long dragons(int actor, Face ron, int count) {
      long yaku = 0;
      int found = 0;
      for (int index = 0; index < DRAGONS.length; index++) {
        int dragon = Tiles.Group.DRAGON.first() + index;
        int tiles = copies.count(actor, dragon) + melded[actor][dragon] + (ron != null && ron.kind() == dragon ? 1 : 0);
        if (tiles >= Meld.Type.PON.tiles()) {
          yaku |= DRAGONS[index].bit();
          found++;
        }
      }
      if (found != count) {
        throw new IllegalArgumentException(id + ": seat " + actor + " has " + found + " dragon triplets, not " + count);
      }
      return yaku;
    }

    /**
     * ryukyoku の終わり方を読む。reason が途中流局ならその流局にし、荒牌なら tenpais から聴牌の席を読む。reason が無いか、荒牌なのに
     * tenpais が無ければ、終わり方を持たない流局にする（局の終わり方は null のまま）。
     */
    private void draw(JsonObject event) {
      if (ended) {
        throw new IllegalArgumentException(id + " ends round " + start.round() + " twice");
      }
      ended = true;
      JsonElement text = optional(event, "reason");
      Reason reason = text == null ? null : Reason.parse(text.getAsString());
      JsonElement tenpais = optional(event, "tenpais");
      if (reason != null && reason.abortive != null) {
        kind = EndKind.ABORTIVE;
        abortive = reason.abortive;
      } else if (reason != null && tenpais != null) {
        kind = EndKind.EXHAUSTIVE;
        nagashi = reason == Reason.NAGASHIMANGAN;
        JsonArray seatsTenpai = tenpais.getAsJsonArray();
        if (seatsTenpai.size() != seats) {
          throw new IllegalArgumentException("mjai tenpais needs " + seats + " values, got " + seatsTenpai.size());
        }
        for (int seat = 0; seat < seats; seat++) {
          tenpaiMask |= seatsTenpai.get(seat).getAsBoolean() ? 1 << seat : 0;
        }
      }
      add(integers(event, "deltas", seats));
    }

    private void add(int[] deltas) {
      for (int seat = 0; seat < seats; seat++) {
        moved[seat] += deltas[seat];
      }
    }

    /** end_kyoku で局の記録を仕上げる。ダブロンの和了は、放銃者から手番の近い順に並べ直す。 */
    private void finishRound() {
      if (!ended) {
        throw new IllegalArgumentException(id + " ends round " + start.round() + " without a hora or ryukyoku");
      }
      int[] deltas = moved.clone();
      for (int seat = 0; seat < seats; seat++) {
        if ((accepted & 1 << seat) != 0) {
          deltas[seat] -= Variant.DEPOSIT;
        }
      }
      RoundResult result;
      if (kind == null) {
        result = RoundResult.drawWithoutDetails(deltas);
      } else {
        wins.sort(Comparator.comparingInt(win -> Math.floorMod(win.seat() - loser, seats)));
        int[] uraIndicators = ura == null ? new int[0] : ura;
        int nagashiSeats = nagashi ? nagashiMask() : 0;
        result = new RoundResult(kind, abortive, wins, loser, tenpaiMask, nagashiSeats, deltas, uraIndicators);
      }
      rounds.add(new RecordedRound(start, deal, indicator, RecordedRound.contractOrder(events), result));
      start = null;
    }

    /** scores の無い end_game のときの終局の点。最後の局の始まりの点に点の移動を足し、場に残った供託をトップに渡して求める。 */
    private int[] settledScores() {
      if (rounds.isEmpty()) {
        throw new IllegalArgumentException(id + " ends the game before its first round");
      }
      RecordedRound last = rounds.getLast();
      int[] scores = last.start().scores();
      int[] deltas = last.result().deltas();
      for (int seat = 0; seat < seats; seat++) {
        scores[seat] += deltas[seat];
      }
      int deposits = last.result().win() ? 0 : last.start().kyotaku() + Integer.bitCount(last.riichiMask());
      Placement.awardDeposits(scores, deposits);
      return scores;
    }

    /**
     * 流し満貫の席を席ごとのビットで返す。荒牌の精算が deltas の和と一致する空でない席の組のうち、最初に見つけたものを使う。一致する
     * 組が無ければ IllegalArgumentException を投げる。
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

    private int seat(int seat) {
      if (seat < 0 || seat >= seats) {
        throw new IllegalArgumentException(id + " names seat " + seat + " of " + seats);
      }
      return seat;
    }
  }

  /** 1 つの牌譜を書く。面子と最後の牌は局ごとに追う。 */
  private static final class Writer {

    private final Variant variant;
    private final TileSet set;
    private final int seats;
    private final StringBuilder lines = new StringBuilder();

    /** 席ごとの面子。加槓は元のポンの位置に置く。 */
    private final List<List<Meld>> melds = new ArrayList<>();

    /** 席ごとに、最後にツモった牌。 */
    private final int[] drawn;

    /** 席ごとに、この局で抜いた北の数。 */
    private final int[] nukis;

    /** チー・ポン・大明槓をした席（席ごとのビット）。 */
    private int openMask;

    private int lastDiscarder;
    private int lastDiscard;

    /** ロンされうる最後の牌（打牌か、加槓で足した牌か、抜いた北）。 */
    private Face claimable;

    Writer(Variant variant) {
      this.variant = variant;
      set = variant.tiles();
      seats = variant.seats();
      drawn = new int[seats];
      nukis = new int[seats];
      for (int seat = 0; seat < seats; seat++) {
        melds.add(new ArrayList<>());
      }
    }

    String write(RecordedGame game) {
      JsonArray names = new JsonArray();
      for (String name : game.meta().names()) {
        names.add(name);
      }
      JsonObject start = event(Type.START_GAME);
      start.add("names", names);
      line(start);
      for (RecordedRound round : game.rounds()) {
        round(round);
      }
      JsonObject end = event(Type.END_GAME);
      end.add("scores", integers(game.finalScores()));
      line(end);
      return lines.toString();
    }

    private void round(RecordedRound round) {
      RoundStart start = round.start();
      for (List<Meld> seatMelds : melds) {
        seatMelds.clear();
      }
      Arrays.fill(drawn, TurnEvent.NONE);
      Arrays.fill(nukis, 0);
      openMask = 0;
      lastDiscarder = TurnEvent.NONE;
      lastDiscard = TurnEvent.NONE;
      claimable = null;
      JsonObject kyoku = event(Type.START_KYOKU);
      kyoku.addProperty("bakaze", Face.of(variant.roundWind(start.round()), false).mjai());
      kyoku.addProperty("kyoku", start.round() % seats + 1);
      kyoku.addProperty("honba", start.honba());
      kyoku.addProperty("kyotaku", start.kyotaku());
      kyoku.addProperty("oya", variant.dealer(start.round()));
      kyoku.add("scores", integers(start.scores()));
      JsonArray hands = new JsonArray();
      for (int seat = 0; seat < seats; seat++) {
        JsonArray hand = new JsonArray();
        for (int index = 0; index < Variant.HAND; index++) {
          hand.add(face(round.deal(seat, index)).mjai());
        }
        hands.add(hand);
      }
      kyoku.add("tehais", hands);
      kyoku.addProperty("dora_marker", face(round.indicator()).mjai());
      line(kyoku);
      List<RecordEvent> events = round.events();
      for (int index = 0; index < events.size(); index++) {
        switch (events.get(index)) {
          case RecordEvent.Draw draw -> {
            JsonObject tsumo = actor(Type.TSUMO, draw.seat());
            tsumo.addProperty("pai", face(draw.tile()).mjai());
            line(tsumo);
            drawn[draw.seat()] = draw.tile();
          }
          case RecordEvent.Discard discard -> index = discard(events, index, discard);
          case RecordEvent.Call call -> call(call);
          case RecordEvent.ClosedKan kan -> {
            Meld meld = Meld.ankan(kan.kind(), redKind(kan.kind()));
            JsonObject ankan = actor(Type.ANKAN, kan.seat());
            ankan.add("consumed", faces(meld, -1));
            line(ankan);
            melds.get(kan.seat()).add(meld);
          }
          case RecordEvent.AddedKan kan -> addedKan(kan);
          case RecordEvent.DoraReveal reveal -> dora(reveal.tile());
          case RecordEvent.RiichiAccepted riichi -> line(actor(Type.REACH_ACCEPTED, riichi.seat()));
          case RecordEvent.Nuki nuki -> {
            JsonObject event = actor(Type.KITA, nuki.seat());
            event.addProperty("pai", face(nuki.tile()).mjai());
            line(event);
            nukis[nuki.seat()]++;
            claimable = face(nuki.tile());
          }
        }
      }
      if (round.result().win()) {
        wins(round);
      } else {
        drawResult(round);
      }
      line(event(Type.END_KYOKU));
    }

    /**
     * 打牌を書く。書く順は、直後にめくる表示牌、reach（立直の宣言牌のときだけ）、打牌。ここで書いた最後のイベント（打牌か、後に
     * 続く表示牌）が、イベントの列の何番目かを返す。
     */
    private int discard(List<RecordEvent> events, int index, RecordEvent.Discard discard) {
      int last = index;
      while (last + 1 < events.size() && events.get(last + 1) instanceof RecordEvent.DoraReveal reveal) {
        dora(reveal.tile());
        last++;
      }
      if (discard.riichi()) {
        line(actor(Type.REACH, discard.seat()));
      }
      JsonObject dahai = actor(Type.DAHAI, discard.seat());
      dahai.addProperty("pai", face(discard.tile()).mjai());
      dahai.addProperty("tsumogiri", discard.tsumogiri());
      line(dahai);
      drawn[discard.seat()] = TurnEvent.NONE;
      lastDiscarder = discard.seat();
      lastDiscard = discard.tile();
      claimable = face(discard.tile());
      return last;
    }

    private void call(RecordEvent.Call call) {
      Meld meld = Meld.of(call.meld());
      JsonObject event = actor(Type.valueOf(meld.type().name()), call.seat());
      event.addProperty("target", lastDiscarder);
      event.addProperty("pai", face(lastDiscard).mjai());
      event.add("consumed", faces(meld, meld.called()));
      line(event);
      melds.get(call.seat()).add(meld);
      openMask |= 1 << call.seat();
    }

    private void addedKan(RecordEvent.AddedKan kan) {
      List<Meld> seatMelds = melds.get(kan.seat());
      for (int index = 0; index < seatMelds.size(); index++) {
        Meld pon = seatMelds.get(index);
        if (pon.type() == Meld.Type.PON && pon.base() == kan.kind()) {
          Face added = Face.of(kan.kind(), redKind(kan.kind()) && !pon.red());
          JsonObject event = actor(Type.KAKAN, kan.seat());
          event.addProperty("pai", added.mjai());
          event.add("consumed", faces(pon, -1));
          line(event);
          seatMelds.set(index, Meld.kakan(pon, added.red()));
          claimable = added;
          return;
        }
      }
      throw new IllegalArgumentException("Seat " + kan.seat() + " has no pon of kind " + kan.kind() + " to add to");
    }

    private void dora(int tile) {
      JsonObject dora = event(Type.DORA);
      dora.addProperty("dora_marker", face(tile).mjai());
      line(dora);
    }

    /**
     * hora を書く（ダブロンは和了者ごとに 1 つ）。2 人目からの和了者の deltas は、本場を付けないロンの精算として Settlement で
     * 計算する。局の点の移動からそれらを引いた残りが、最初の和了者の deltas になる。内訳を持たない和了の点の移動は最初の和了者の分に
     * 含め、その hora には内訳のキー（yakus・fu・fan・hora_points・pao）を書かない。
     */
    private void wins(RecordedRound round) {
      RoundResult result = round.result();
      int dealer = variant.dealer(round.start().round());
      boolean tsumo = result.kind() == EndKind.TSUMO;
      List<RoundResult.Win> wins = result.wins();
      int[][] parts = new int[wins.size()][seats];
      parts[0] = moved(round);
      for (int index = 1; index < wins.size(); index++) {
        RoundResult.Win win = wins.get(index);
        if (!win.scored()) {
          continue;
        }
        Settlement.ron(variant, win.base(), win.seat(), result.loser(), dealer, 0, win.pao(), parts[index]);
        for (int seat = 0; seat < seats; seat++) {
          parts[0][seat] -= parts[index][seat];
        }
      }
      for (int index = 0; index < wins.size(); index++) {
        RoundResult.Win win = wins.get(index);
        boolean riichi = (round.riichiMask() & 1 << win.seat()) != 0;
        JsonObject hora = actor(Type.HORA, win.seat());
        hora.addProperty("target", tsumo ? win.seat() : result.loser());
        hora.addProperty("pai", (tsumo ? face(drawn[win.seat()]) : claimable).mjai());
        if (riichi) {
          JsonArray markers = new JsonArray();
          for (int tile : result.uraIndicators()) {
            markers.add(face(tile).mjai());
          }
          hora.add("uradora_markers", markers);
        }
        if (win.scored()) {
          hora.add("yakus", yakus(win, riichi));
          hora.addProperty("fu", win.fu());
          hora.addProperty("fan", win.han() > 0 ? win.han() : win.base() / YAKUMAN_BASE * YAKUMAN_FAN);
          hora.addProperty("hora_points", win.points(variant, dealer, tsumo));
          if (win.pao() != RoundOutcome.NO_SEAT) {
            hora.addProperty("pao", win.pao());
          }
        }
        hora.add("deltas", integers(parts[index]));
        line(hora);
      }
    }

    /** ryukyoku を書く。終わり方を持たない流局は、reason・tenpais を書かない。 */
    private void drawResult(RecordedRound round) {
      RoundResult result = round.result();
      JsonObject ryukyoku = event(Type.RYUKYOKU);
      if (result.kind() != null) {
        ryukyoku.addProperty("reason", Reason.of(result).text());
        JsonArray tenpais = new JsonArray();
        for (int seat = 0; seat < seats; seat++) {
          tenpais.add((result.tenpaiMask() & 1 << seat) != 0);
        }
        ryukyoku.add("tenpais", tenpais);
      }
      ryukyoku.add("deltas", integers(moved(round)));
      line(ryukyoku);
    }

    /**
     * 役と翻の組を、Yaku の宣言の順に並べる。三元牌の役は sangenpai に刻子の数としてまとめ、役満は役 1 つにつき 100 とする。記録の
     * ドラの翻のうち、和了者がこの局で抜いた北の数（ドラの翻を超えない）を nukidora に分けて書く。役満の和了はドラの翻を持たない。
     * 裏ドラは、立直した和了なら 0 翻でも書く。役満でない和了で翻の和が記録と違えば IllegalArgumentException を投げる。
     */
    private JsonArray yakus(RoundResult.Win win, boolean riichi) {
      boolean closed = (openMask & 1 << win.seat()) == 0;
      int[] han = new int[Yaku.values().length];
      for (ScoringYaku yaku : ScoringYaku.values()) {
        if ((win.yaku() & yaku.bit()) != 0) {
          han[Yaku.of(yaku).ordinal()] += yaku.yakuman() ? YAKUMAN_FAN : yaku.han(closed);
        }
      }
      int nuki = Math.min(nukis[win.seat()], win.dora());
      han[Yaku.DORA.ordinal()] = win.dora() - nuki;
      han[Yaku.NUKIDORA.ordinal()] = nuki;
      han[Yaku.AKADORA.ordinal()] = win.aka();
      han[Yaku.URADORA.ordinal()] = win.ura();
      JsonArray yakus = new JsonArray();
      int total = 0;
      for (Yaku yaku : Yaku.values()) {
        if (han[yaku.ordinal()] > 0 || yaku == Yaku.URADORA && riichi) {
          JsonArray pair = new JsonArray();
          pair.add(yaku.text());
          pair.add(han[yaku.ordinal()]);
          yakus.add(pair);
          total += han[yaku.ordinal()];
        }
      }
      if (win.han() > 0 && total != win.han()) {
        throw new IllegalArgumentException("The yaku of " + win + " do not add up to its han");
      }
      return yakus;
    }

    /** 記録の点の移動に、この局の立直の供託を戻したもの（mjai の deltas は立直の供託を含まない）。 */
    private int[] moved(RecordedRound round) {
      int[] moved = round.result().deltas();
      for (int seat = 0; seat < seats; seat++) {
        if ((round.riichiMask() & 1 << seat) != 0) {
          moved[seat] += Variant.DEPOSIT;
        }
      }
      return moved;
    }

    /**
     * 面子の牌を mjai の表記で並べる。鳴いた牌を 1 枚除いて書くか、全ての牌を書く。赤は 1 枚まで書く。赤を書くのは、鳴いた牌を
     * 除くときは手から出した牌に赤があるとき、全ての牌を書くときは面子に赤があるとき。
     *
     * @param called 鳴いた牌の種類で、その 1 枚を除く。−1 なら面子の全ての牌を書く
     */
    private JsonArray faces(Meld meld, int called) {
      JsonArray faces = new JsonArray();
      boolean red = called < 0 ? meld.red() : meld.aka() == Meld.AkaSource.CONSUMED_HAND_TILE;
      boolean calledSkipped = called < 0;
      for (int index = 0; index < meld.size(); index++) {
        int kind = meld.tile(index);
        if (!calledSkipped && kind == called) {
          calledSkipped = true;
          continue;
        }
        boolean redHere = red && redKind(kind);
        red &= !redHere;
        faces.add(Face.of(kind, redHere).mjai());
      }
      return faces;
    }

    /** その種類に赤があるか。 */
    private boolean redKind(int kind) {
      return set.red(Tiles.physicalId(kind, Tiles.RED_COPY));
    }

    private Face face(int tile) {
      return Face.of(Tiles.kind(tile), set.red(tile));
    }

    private JsonObject actor(Type type, int seat) {
      JsonObject event = event(type);
      event.addProperty("actor", seat);
      return event;
    }

    private static JsonObject event(Type type) {
      JsonObject event = new JsonObject();
      event.addProperty("type", type.text());
      return event;
    }

    private static JsonArray integers(int[] values) {
      JsonArray array = new JsonArray();
      for (int value : values) {
        array.add(value);
      }
      return array;
    }

    private void line(JsonObject event) {
      lines.append(event).append(LINE);
    }
  }
}
