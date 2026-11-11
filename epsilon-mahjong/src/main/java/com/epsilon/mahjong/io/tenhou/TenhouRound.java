package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * 天鳳の局のタグ（INIT・REACH・AGARI・RYUUKYOKU）を読む。牌譜とライブの両方で使う。タグの席は回転させて絶対席に直す（絶対席 =
 * (タグの席 ＋ 回転) mod 席の数）。回転は INIT の親から決まり、牌譜では 0、自分を席 0 と書くライブでは自分の絶対席になる。
 *
 * <p>局の結果は、AGARI（ダブロンなら続けて複数）か RYUUKYOKU を積んで作る。AGARI の点（ten）が、翻・符か役満から計算した和了点と
 * 違えば IllegalArgumentException を投げる。局の点の移動は、sc の移動を足し、この局で成立した立直の供託を引いて求める。流し満貫の
 * 席は、荒牌の精算が sc の移動と一致する席の組にする。1 つのインスタンスを同時に使えるのは 1 つのスレッドだけ。
 *
 * <p>天鳳は席ごとの値（ten・sc・owari、INIT の配牌）を、三人打ちでも 4 席分書く。ルールの席の数より後ろの席は空き席で、値は 0 か
 * 空になる。読むときは、空き席の値が 0 であることを確かめてから捨てる。
 */
public final class TenhouRound {

  /** ten・sc・owari の点の単位。 */
  static final int POINT_UNIT = 100;

  /** 天鳳のタグが席ごとの値を並べる席の数（空き席を含む）。 */
  static final int TAG_SEATS = 4;

  /** INIT の seed の値の並び（宣言の順）。 */
  enum Seed {
    ROUND,
    HONBA,
    KYOTAKU,
    DIE,
    OTHER_DIE,
    INDICATOR
  }

  /** AGARI の ten の値の並び。LIMIT は満貫未満 0、満貫 1 … 三倍満 4、役満 5。 */
  enum Ten {
    FU,
    POINTS,
    LIMIT
  }

  /** sc・owari で 1 席分を表す 2 つの値の順。sc の CHANGE は点の移動、owari の CHANGE は順位の点（千点単位で小数 1 桁）。 */
  enum Pair {
    SCORE,
    CHANGE
  }

  /** REACH の step。値は宣言の順に 1 から振る。 */
  public enum ReachStep {
    DECLARE,
    ACCEPT;

    int value() {
      return ordinal() + 1;
    }

    /** REACH の step を読む。1・2 でなければ IllegalArgumentException を投げる。 */
    public static ReachStep of(TenhouTag reach) {
      int value = reach.integer("step");
      if (value < 1 || value > values().length) {
        throw new IllegalArgumentException("Not a Tenhou REACH step: " + value);
      }
      return values()[value - 1];
    }
  }

  /** RYUUKYOKU の type（定数の名前の小文字）。type の無い RYUUKYOKU は荒牌。 */
  enum DrawType {
    YAO9(AbortiveDraw.NINE_TERMINALS),
    REACH4(AbortiveDraw.FOUR_RIICHI),
    RON3(AbortiveDraw.TRIPLE_RON),
    KAN4(AbortiveDraw.FOUR_KANS),
    KAZE4(AbortiveDraw.FOUR_WINDS),
    /** 流し満貫（荒牌）。 */
    NM(null);

    private final AbortiveDraw abortive;

    DrawType(AbortiveDraw abortive) {
      this.abortive = abortive;
    }

    String text() {
      return name().toLowerCase(Locale.ROOT);
    }

    static DrawType parse(String text) {
      for (DrawType type : values()) {
        if (type.text().equals(text)) {
          return type;
        }
      }
      throw new IllegalArgumentException("Unknown Tenhou RYUUKYOKU type " + text);
    }

    static DrawType of(AbortiveDraw draw) {
      for (DrawType type : values()) {
        if (type.abortive == draw) {
          return type;
        }
      }
      throw new IllegalArgumentException("Tenhou has no RYUUKYOKU type for " + draw);
    }
  }

  /** AGARI の yaku・yakuman に書く役の番号（宣言の順に 0 から）。ドラ・裏ドラ・赤ドラ・抜きドラは翻の数だけを持つ。 */
  enum Yaku {
    MENZEN_TSUMO,
    RIICHI,
    IPPATSU,
    CHANKAN,
    RINSHAN,
    HAITEI,
    HOUTEI,
    PINFU,
    TANYAO,
    IPEIKOU,
    SEAT_TON(ScoringYaku.YAKUHAI_SEAT),
    SEAT_NAN(ScoringYaku.YAKUHAI_SEAT),
    SEAT_SHA(ScoringYaku.YAKUHAI_SEAT),
    SEAT_PEI(ScoringYaku.YAKUHAI_SEAT),
    ROUND_TON(ScoringYaku.YAKUHAI_ROUND),
    ROUND_NAN(ScoringYaku.YAKUHAI_ROUND),
    ROUND_SHA(ScoringYaku.YAKUHAI_ROUND),
    ROUND_PEI(ScoringYaku.YAKUHAI_ROUND),
    YAKUHAI_HAKU,
    YAKUHAI_HATSU,
    YAKUHAI_CHUN,
    DOUBLE_RIICHI,
    CHIITOITSU,
    CHANTA,
    ITTSU,
    SANSHOKU,
    SANSHOKU_DOUKOU,
    SANKANTSU,
    TOITOI,
    SANANKOU,
    SHOUSANGEN,
    HONROUTOU,
    RYANPEIKOU,
    JUNCHAN,
    HONITSU,
    CHINITSU,
    /** 天鳳のルールに無い役。 */
    RENHOU(null),
    TENHOU,
    CHIIHOU,
    DAISANGEN,
    SUUANKOU,
    SUUANKOU_TANKI(ScoringYaku.SUUANKOU),
    TSUUIISOU,
    RYUUIISOU,
    CHINROUTOU,
    CHUUREN,
    JUNSEI_CHUUREN(ScoringYaku.CHUUREN),
    KOKUSHI,
    KOKUSHI_13(ScoringYaku.KOKUSHI),
    DAISUUSHII,
    SHOUSUUSHII,
    SUUKANTSU,
    DORA(null),
    URA(null),
    AKA(null),
    /** 天鳳は抜きドラをドラの翻に含めて書く。 */
    NUKI(null);

    static final Yaku[] VALUES = values();

    /** 役を書くときの番号を、ScoringYaku の ordinal で引く表。同じ役に番号が複数あれば最も小さい番号を入れる。 */
    private static final Yaku[] OF_SCORING = ofScoring();

    private final ScoringYaku scoring;

    Yaku() {
      scoring = ScoringYaku.valueOf(name());
    }

    Yaku(ScoringYaku scoring) {
      this.scoring = scoring;
    }

    static Yaku of(int number) {
      if (number < 0 || number >= VALUES.length) {
        throw new IllegalArgumentException("Unknown Tenhou yaku " + number);
      }
      return VALUES[number];
    }

    /** 役を書くときの番号を返す。自風・場風の役は風ごとの番号、ほかは同じ役の番号のうち最も小さいもの。 */
    static int number(ScoringYaku yaku, Variant variant, int round, int seat) {
      return switch (yaku) {
        case YAKUHAI_SEAT -> SEAT_TON.ordinal() + variant.seatWind(round, seat) - Tiles.TON;
        case YAKUHAI_ROUND -> ROUND_TON.ordinal() + variant.roundWind(round) - Tiles.TON;
        default -> OF_SCORING[yaku.ordinal()].ordinal();
      };
    }

    ScoringYaku scoring() {
      if (scoring == null) {
        throw new IllegalArgumentException("Tenhou yaku " + this + " is not a yaku of the Tenhou rules");
      }
      return scoring;
    }

    private static Yaku[] ofScoring() {
      Yaku[] table = new Yaku[ScoringYaku.values().length];
      for (Yaku yaku : VALUES) {
        if (yaku.scoring != null && table[yaku.scoring.ordinal()] == null) {
          table[yaku.scoring.ordinal()] = yaku;
        }
      }
      return table;
    }
  }

  private final Variant variant;
  private final TileSet set;
  private final int seats;
  private final List<RoundResult.Win> wins = new ArrayList<>();

  private RoundStart start;
  private int rotation;
  private int indicator;
  private EndKind kind;
  private AbortiveDraw abortive;
  private boolean nagashi;
  private int loser;
  private int tenpaiMask;

  /** 絶対席ごとに足し上げた sc の移動。 */
  private int[] moved;

  /** 裏ドラ表示牌の物理 id（AGARI の doraHaiUra）。 */
  private int[] ura;

  /** owari の終局の点（絶対席ごと）。読むまで null。 */
  private int[] finalScores;

  public TenhouRound(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    seats = variant.seats();
  }

  /**
   * INIT を読んで局を始め、局の始まりの卓を返す。回転は、親の絶対席から oya を引いて決める。読めない INIT なら
   * IllegalArgumentException を投げる。
   */
  public RoundStart start(TenhouTag init) {
    int[] seed = init.integers("seed");
    if (seed.length != Seed.values().length) {
      throw new IllegalArgumentException("Tenhou INIT has a seed of " + seed.length + " values");
    }
    int round = TenhouRoundIndex.round(seed[Seed.ROUND.ordinal()], variant);
    int oya = init.integer("oya");
    if (oya < 0 || oya >= seats) {
      throw new IllegalArgumentException("Tenhou INIT names dealer " + oya + " of " + seats);
    }
    rotation = Math.floorMod(variant.dealer(round) - oya, seats);
    int[] ten = init.integers("ten");
    if (ten.length != TAG_SEATS) {
      throw new IllegalArgumentException("Tenhou INIT has " + ten.length + " scores for " + TAG_SEATS + " seats");
    }
    int[] scores = new int[seats];
    for (int tagSeat = 0; tagSeat < TAG_SEATS; tagSeat++) {
      if (tagSeat < seats) {
        scores[seat(tagSeat)] = ten[tagSeat] * POINT_UNIT;
      } else {
        requireEmpty("INIT ten", tagSeat, ten[tagSeat]);
      }
    }
    start = new RoundStart(variant, round, seed[Seed.HONBA.ordinal()], seed[Seed.KYOTAKU.ordinal()], scores);
    indicator = TenhouTiles.tile(seed[Seed.INDICATOR.ordinal()], set);
    wins.clear();
    kind = null;
    abortive = null;
    nagashi = false;
    loser = RoundOutcome.NO_SEAT;
    tenpaiMask = 0;
    moved = new int[seats];
    ura = new int[0];
    return start;
  }

  /** タグの席 0 が何番の絶対席か（回転）。 */
  public int rotation() {
    return rotation;
  }

  /** INIT の最初のドラ表示牌の物理 id。 */
  public int indicator() {
    return indicator;
  }

  /** タグの席を絶対席に直す。そのルールに無い席なら IllegalArgumentException を投げる。 */
  public int seat(int tagSeat) {
    if (tagSeat < 0 || tagSeat >= seats) {
      throw new IllegalArgumentException("Tenhou names seat " + tagSeat + " of " + seats);
    }
    return (tagSeat + rotation) % seats;
  }

  /** AGARI を積む。ダブロンなら、同じ放銃者からのロンの AGARI を続けて積む。 */
  public void agari(TenhouTag agari) {
    requireStarted();
    int who = seat(agari.integer("who"));
    int from = seat(agari.integer("fromWho"));
    boolean tsumo = who == from;
    if (kind != null && (tsumo || kind != EndKind.RON || from != loser)) {
      throw new IllegalArgumentException("Round " + start.round() + " ends twice");
    }
    int[] ten = agari.integers("ten");
    if (ten.length != Ten.values().length) {
      throw new IllegalArgumentException("Tenhou AGARI has a ten of " + ten.length + " values");
    }
    RoundResult.Win win = scored(agari, who, ten[Ten.FU.ordinal()]);
    int points = win.points(variant, variant.dealer(start.round()), tsumo);
    if (points != ten[Ten.POINTS.ordinal()]) {
      throw new IllegalArgumentException(
          "The AGARI of seat " + who + " pays " + ten[Ten.POINTS.ordinal()] + ", but " + win + " pays " + points);
    }
    wins.add(win);
    kind = tsumo ? EndKind.TSUMO : EndKind.RON;
    loser = tsumo ? RoundOutcome.NO_SEAT : from;
    String uraIndicators = agari.get("doraHaiUra");
    if (uraIndicators != null) {
      ura = TenhouTiles.tiles(uraIndicators, set);
    }
    settle(agari);
  }

  /** RYUUKYOKU を積む。 */
  public void ryuukyoku(TenhouTag ryuukyoku) {
    requireStarted();
    if (kind != null) {
      throw new IllegalArgumentException("Round " + start.round() + " ends twice");
    }
    String text = ryuukyoku.get("type");
    DrawType type = text == null ? null : DrawType.parse(text);
    if (type == null || type.abortive == null) {
      kind = EndKind.EXHAUSTIVE;
      nagashi = type == DrawType.NM;
      for (int tagSeat = 0; tagSeat < seats; tagSeat++) {
        if (ryuukyoku.get("hai" + tagSeat) != null) {
          tenpaiMask |= 1 << seat(tagSeat);
        }
      }
    } else {
      kind = EndKind.ABORTIVE;
      abortive = type.abortive;
    }
    settle(ryuukyoku);
  }

  /**
   * 積んだタグから局の結果を作る。
   *
   * @param riichiMask この局で立直が成立した席（席ごとのビット）
   * @throws IllegalArgumentException AGARI も RYUUKYOKU も読んでいないとき
   */
  public RoundResult result(int riichiMask) {
    requireStarted();
    if (kind == null) {
      throw new IllegalArgumentException("Round " + start.round() + " has no AGARI or RYUUKYOKU");
    }
    int[] deltas = moved.clone();
    for (int seat = 0; seat < seats; seat++) {
      if ((riichiMask & 1 << seat) != 0) {
        deltas[seat] -= Variant.DEPOSIT;
      }
    }
    return new RoundResult(kind, abortive, wins, loser, tenpaiMask, nagashi ? nagashiMask() : 0, deltas, ura);
  }

  /** owari の終局の点（絶対席ごと、配列のコピー）。owari を読んでいなければ null。 */
  public int[] finalScores() {
    return finalScores == null ? null : finalScores.clone();
  }

  /**
   * AGARI の役・翻・符から和了を作る。役満なら、役満の役の数が倍率になる。
   *
   * @param who 和了した席（絶対席）
   */
  private RoundResult.Win scored(TenhouTag agari, int who, int fu) {
    long yaku = 0;
    int han = 0;
    int dora = 0;
    int aka = 0;
    int uraHan = 0;
    int yakuman = 0;
    if (agari.get("yakuman") != null) {
      for (int number : agari.integers("yakuman")) {
        yaku |= Yaku.of(number).scoring().bit();
        yakuman++;
      }
    } else {
      int[] pairs = agari.integers("yaku");
      if (pairs.length % 2 != 0) {
        throw new IllegalArgumentException("Tenhou AGARI has a yaku list of odd length");
      }
      for (int index = 0; index < pairs.length; index += 2) {
        Yaku each = Yaku.of(pairs[index]);
        int value = pairs[index + 1];
        han += value;
        switch (each) {
          case DORA, NUKI -> dora += value;
          case URA -> uraHan += value;
          case AKA -> aka += value;
          default -> yaku |= each.scoring().bit();
        }
      }
    }
    int pao = agari.get("paoWho") == null ? RoundOutcome.NO_SEAT : seat(agari.integer("paoWho"));
    return RoundResult.Win.of(who, yaku, han, fu, dora, aka, uraHan, yakuman, pao);
  }

  /** sc の移動を足し、owari があれば終局の点を読む。 */
  private void settle(TenhouTag end) {
    int[] change = pairs(end, "sc", Pair.CHANGE);
    for (int seat = 0; seat < seats; seat++) {
      moved[seat] += change[seat];
    }
    if (end.get("owari") != null) {
      finalScores = pairs(end, "owari", Pair.SCORE);
    }
  }

  /**
   * sc か owari は、席ごとに 2 つの値の組を並べる。各組から片方の値を読み、点に直して絶対席の順に返す。空き席の値が 0 でなければ
   * IllegalArgumentException を投げる。
   */
  private int[] pairs(TenhouTag end, String key, Pair field) {
    String[] values = end.required(key).split(",", -1);
    int width = Pair.values().length;
    if (values.length != TAG_SEATS * width) {
      throw new IllegalArgumentException(
          "Tenhou " + key + " lists " + values.length + " values for " + TAG_SEATS + " seats: " + end.get(key));
    }
    int[] points = new int[seats];
    for (int tagSeat = 0; tagSeat < TAG_SEATS; tagSeat++) {
      int value = Integer.parseInt(values[tagSeat * width + field.ordinal()].trim());
      if (tagSeat < seats) {
        points[seat(tagSeat)] = value * POINT_UNIT;
      } else {
        requireEmpty(end.name() + " " + key, tagSeat, value);
      }
    }
    return points;
  }

  /** 空き席の値が 0 であることを確かめる。0 でなければ IllegalArgumentException を投げる。 */
  private static void requireEmpty(String field, int tagSeat, int value) {
    if (value != 0) {
      throw new IllegalArgumentException("Tenhou " + field + " gives " + value + " to the empty seat " + tagSeat);
    }
  }

  /**
   * 流し満貫の席を席ごとのビットで返す。荒牌の精算が sc の移動と一致する空でない席の組のうち、最初に見つけたものを使う。一致する組が
   * 無ければ IllegalArgumentException を投げる。
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
    throw new IllegalArgumentException("No nagashi mangan pays " + Arrays.toString(moved));
  }

  private void requireStarted() {
    if (start == null) {
      throw new IllegalArgumentException("A Tenhou round tag comes before INIT");
    }
  }
}
