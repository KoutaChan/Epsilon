package com.epsilon.classic.input;

import com.epsilon.classic.Architecture;
import com.epsilon.classic.ArchitectureScoped;
import com.epsilon.classic.input.ClassicVocabulary.ActionGroup;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.ClassicVocabulary.DiscardContext;
import com.epsilon.classic.input.ClassicVocabulary.DrawSource;
import com.epsilon.classic.input.ClassicVocabulary.EventType;
import com.epsilon.classic.input.ClassicVocabulary.KanKind;
import com.epsilon.classic.input.ClassicVocabulary.MeldAkaSource;
import com.epsilon.classic.input.ClassicVocabulary.MeldSource;
import com.epsilon.classic.input.ClassicVocabulary.MeldType;
import com.epsilon.classic.input.ClassicVocabulary.NextStep;
import com.epsilon.classic.input.ClassicVocabulary.RiichiStatus;
import com.epsilon.classic.input.ClassicVocabulary.RonFuritenKind;
import com.epsilon.classic.input.ClassicVocabulary.SettlementAssumption;
import com.epsilon.classic.input.ClassicVocabulary.TileSelection;
import com.epsilon.classic.input.ClassicVocabulary.TransitionKind;
import com.epsilon.classic.input.ClassicVocabulary.WinConditions;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Set;

/**
 * 入力のフィールド。変換した重み（埋め込みの表と、線形の層が受け取る入力の並び）は名前・並び・幅を前提にしているので、テストで固定する。
 * 各フィールドの値は {@link ClassicTokenizer} が判断の記録から求める。
 *
 * <p>整数のフィールドの値は 0 以上で、幅より小さい。番号・席・種類・enum の値は「値 ＋ 1」で書き、0 はパディングに使う。数と真偽値
 * （偽は 0）はそのまま書く。実数のフィールドには、量を尺度で割った値を書く。パディング（存在しない合法手・遷移・待ち・河・面子の
 * トークン）はどのフィールドも 0 なので、実在するかどうかは番号か {@code PRESENT} のフィールドで見分ける。
 *
 * <p>フィールドのオフセットは、{@link ClassicSchema} が宣言の順から一度だけ求める。アーキテクチャが持たないフィールドは並びに入れない。
 * 席は、断りが無ければ判断する席から見た相対席（0 自分、1 下家、2 対面、3 上家）で書く。
 */
public final class ClassicFields {

  private static final int FLAG = 2;
  private static final int SEAT = ClassicSchema.SEATS + 1;
  private static final int TILE = Tiles.KINDS + 1;
  private static final int COPIES = Tiles.COPIES + 1;
  private static final int KYOKU = 256;
  private static final int STICKS = 512;

  /** 山の残り枚数の幅（牌の総数 − 王牌 ＋ 1）。 */
  private static final int WALL = Tiles.PHYSICAL_IDS - Variant.DEAD_WALL + 1;

  private static final int TURN = 128;
  private static final int KANS = Variant.MAX_KANS + 1;
  private static final int INDICATORS = WinContext.MAX_INDICATORS + 1;
  private static final int MELDS = ClassicSchema.MELD_TOKENS + 1;
  private static final int RIVER = ClassicSchema.RIVER_TOKENS + 1;
  private static final int SEQUENCE = ClassicSchema.SEATS * ClassicSchema.RIVER_TOKENS + 1;
  private static final int CHI_POSITION = ClassicVocabulary.RUN + 1;
  private static final int SLOTS = ClassicSchema.MAX_LEGAL + 1;

  /** 和了の翻の幅（翻は 127 を超えない）。 */
  private static final int HAN = 128;

  /** 符のコードの数（0〜11）。 */
  private static final int FU_CODES = 12;

  /** 役満の倍率の幅（役満の役 12 個が全て重なるまで）。 */
  private static final int YAKUMAN = 13;

  private static final float POINTS = 100_000f;
  private static final float RIVER_SCALE = ClassicSchema.RIVER_TOKENS;
  private static final float WALL_SCALE = 70f;
  private static final float SHANTEN = 9f;
  private static final float LIVE = Tiles.KINDS * Tiles.COPIES;
  private static final float KIND_SCALE = Tiles.KINDS;
  private static final float HAN_SCALE = 13f;
  private static final float FU_SCALE = 110f;
  private static final float BASE_SCALE = 32_000f;

  private ClassicFields() {}

  /** enum の値 ＋ 1 を書くフィールドの幅（enum の値の数 ＋ 1）。 */
  private static int vocabulary(Class<? extends Enum<?>> type) {
    return type.getEnumConstants().length + 1;
  }

  /** 局と直前のイベントを表す整数のフィールド（1 行に 1 つずつ）。 */
  public enum RoundInt implements CategoricalField {
    /** 判断する席の絶対席 ＋ 1。 */
    PLAYER_SEAT(SEAT),
    /** 手番の席 ＋ 1。自分の手番とチー・ポンの後の打牌では自分、他家への応答では直前のイベントの席。 */
    CURRENT_PLAYER_RELATIVE_SEAT(SEAT),
    /** 直前のイベントの席 ＋ 1。イベントが無ければ自分（1）。 */
    SOURCE_PLAYER_RELATIVE_SEAT(SEAT),
    /** 局の通し番号（東 1 局が 0）＋ 1。 */
    KYOKU_INDEX(KYOKU),
    /** 親の席 ＋ 1。 */
    DEALER_RELATIVE_SEAT(SEAT),
    /** 場風の種類 ＋ 1。 */
    BAKAZE(TILE),
    /** 自風の種類 ＋ 1。 */
    JIKAZE(TILE),
    /** 本場の数。 */
    HONBA(STICKS),
    /** 供託の本数。 */
    KYOTAKU(STICKS),
    /** 王牌を除いた山の残り枚数。 */
    WALL_REMAINING(WALL),
    /** 巡目。ツモの後の打牌の数を席の数で割り、切り捨てる（チー・ポンの後の打牌は数えない）。 */
    TURN_NUMBER(TURN),
    /** 全ての席の槓子の数。 */
    TOTAL_KAN(KANS),
    /** ドラ表示の数。 */
    DORA_INDICATOR_COUNT(INDICATORS),
    /** オーラス（南 4 局）か、それより後の局なら 1。 */
    ALL_LAST(FLAG),
    /** まだ 1 枚も打っていない席があれば 1。 */
    INITIAL_DISCARD_CYCLE(FLAG),
    /** 自分がまだ 1 枚も打っていなければ 1。 */
    SELF_BEFORE_FIRST_DISCARD(FLAG),
    /** 自分がまだ 1 枚も打っておらず、局に副露（暗槓を含む）も無ければ 1。 */
    UNINTERRUPTED_FIRST_DRAW(FLAG),
    /** 局に副露（暗槓を含む）があれば 1。 */
    FIRST_TURN_CALL_OCCURRED(FLAG),
    /** 山の残りが 0 なら 1。 */
    LAST_LIVE_TILE(FLAG),
    /** 直前のイベントの種類 ＋ 1（{@link EventType} の順）。 */
    EVENT_TYPE(vocabulary(EventType.class)),
    /** 直前のイベントが槓の宣言なら、槓の種類 ＋ 1（{@link KanKind} の順）。ほかは 0。 */
    EVENT_KAN_KIND(vocabulary(KanKind.class)),
    /** 直前のイベントの席 ＋ 1。イベントが無ければ 0。 */
    EVENT_PLAYER_RELATIVE_SEAT(SEAT),
    /** 直前のイベントの牌の種類 ＋ 1。イベントが無いときと、他家のツモでは 0。 */
    EVENT_TILE(TILE),
    /** 直前のイベントの牌が赤なら 1。 */
    EVENT_IS_AKA(FLAG),
    /** 直前のイベントがツモなら、どこから引いたか ＋ 1（{@link DrawSource} の順）。ほかは 0。 */
    EVENT_DRAW_SOURCE(vocabulary(DrawSource.class)),
    /** 直前のイベントが嶺上牌のツモなら 1。 */
    EVENT_RINSHAN(FLAG),
    /** 明槓・加槓の後で、槓ドラの表示をまだめくっていなければ 1。 */
    EVENT_DEFERRED_DORA(FLAG);

    private final int width;
    private final Set<Architecture> architectures;

    RoundInt(int width, Architecture... only) {
      this.width = width;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 局と点差を表す実数のフィールド（1 行に 1 つずつ）。 */
  public enum RoundFloat implements NumericField {
    /** 本場の数 / 10。 */
    HONBA(10),
    /** 供託の本数 / 10。 */
    KYOTAKU(10),
    /** 山の残り / 70。 */
    WALL_REMAINING(WALL_SCALE),
    /** 巡目 / 18。 */
    TURN(18),
    /** 自分の点 / 100000。 */
    SELF_SCORE(POINTS),
    /** (自分の点 − 他家の最高点) / 100000。 */
    SCORE_LEAD(POINTS),
    /** (自分の点 − 最高点) / 100000。 */
    SCORE_TO_FIRST(POINTS),
    /** (自分の点 − 最低点) / 100000。 */
    SCORE_TO_FOURTH(POINTS);

    private final float scale;
    private final Set<Architecture> architectures;

    RoundFloat(float scale, Architecture... only) {
      this.scale = scale;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 席ごとの整数のフィールド（相対席の順に 4 つ）。 */
  public enum PlayerInt implements CategoricalField {
    /** 相対席 ＋ 1。 */
    RELATIVE_SEAT(SEAT),
    /** 絶対席 ＋ 1。 */
    ABSOLUTE_SEAT(SEAT),
    /** 自風の種類 ＋ 1。 */
    JIKAZE(TILE),
    /** 立直の状態 ＋ 1（{@link RiichiStatus} の順）。宣言牌がまだ通っていない立直は、宣言牌が通ったときの状態を書く。 */
    RIICHI_STATUS(vocabulary(RiichiStatus.class)),
    /** 一発の権利があれば 1。宣言牌がまだ通っていない立直も 1。 */
    IPPATSU(FLAG),
    /** まだ 1 枚も打っていなければ 1。 */
    BEFORE_FIRST_DISCARD(FLAG),
    /** 自分の席だけに書く。見逃しによる振聴（同巡内と立直の後）なら 1。 */
    SELF_TEMPORARY_FURITEN(FLAG),
    /** 自分の席だけに書く。今の手で、形の待ちのどれかが自分の河にあれば 1。 */
    SELF_PERMANENT_FURITEN(FLAG),
    /** 門前を崩す面子が無ければ 1。 */
    MENZEN(FLAG),
    /** 点の順位（0 始まり、同点は起家に近い席が上）＋ 1。 */
    RANK(SEAT),
    /** 面子（副露と暗槓）の数。 */
    MELD_COUNT(MELDS),
    /** 門前を崩す面子の数。 */
    OPEN_MELD_COUNT(MELDS),
    /** 槓子の数。 */
    KAN_COUNT(MELDS),
    /** 河に打った牌の数（鳴かれた牌を含む）。24 を上限にする。 */
    RIVER_COUNT(RIVER),
    /** 立直の宣言牌の河での位置（0 始まり）＋ 1。24 を上限にし、立直していなければ 0。 */
    RIICHI_DECLARATION_INDEX(RIVER),
    /** 立直の宣言牌より後に、その席が打った牌の数。24 を上限にする。 */
    DISCARDS_AFTER_RIICHI(RIVER),
    /** そのうちツモ切りした数。24 を上限にする。 */
    POST_RIICHI_TSUMOGIRI_COUNT(RIVER);

    private final int width;
    private final Set<Architecture> architectures;

    PlayerInt(int width, Architecture... only) {
      this.width = width;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 席ごとの実数のフィールド（相対席の順に 4 つ）。 */
  public enum PlayerFloat implements NumericField {
    /** 点 / 100000。 */
    SCORE(POINTS),
    /** (その席の点 − 自分の点) / 100000。 */
    SCORE_FROM_SELF(POINTS),
    /** 河に打った牌の数 / 24。 */
    RIVER_PROGRESS(RIVER_SCALE),
    /** 門前を崩す面子の数 / 4。 */
    OPEN_MELD_FRACTION(ClassicSchema.MELD_TOKENS),
    /** (立直の宣言牌の河での位置 ＋ 1) / 24。位置は 0 始まりで、立直していなければ 0。 */
    RIICHI_DECLARATION_PROGRESS(RIVER_SCALE),
    /** 親なら 1。 */
    DEALER(1);

    private final float scale;
    private final Set<Architecture> architectures;

    PlayerFloat(float scale, Architecture... only) {
      this.scale = scale;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 牌の種類ごとの整数のフィールド（種類の順に 34 個）。 */
  public enum TileInt implements CategoricalField {
    /** 種類 ＋ 1。 */
    TILE_TYPE(TILE),
    /** 自分の純手牌の枚数。 */
    SELF_HAND_COUNT(COPIES),
    /** 自分の純手牌にその種類の赤があれば 1。 */
    SELF_HAS_AKA(FLAG),
    /** 見えている枚数（河・全ての席の面子・ドラ表示。自分の純手牌は含まない）。 */
    VISIBLE_COUNT(COPIES),
    /** その種類の牌がドラ表示に出ている数。 */
    DORA_INDICATOR_MULTIPLICITY(INDICATORS),
    /** その種類をドラにするドラ表示の数。 */
    DORA_MULTIPLICITY(INDICATORS),
    /** 自分の河にある枚数（鳴かれた牌を含む）。 */
    SELF_RIVER_COUNT(COPIES),
    /** 下家の河にある枚数。 */
    SHIMOCHA_RIVER_COUNT(COPIES),
    /** 対面の河にある枚数。 */
    TOIMEN_RIVER_COUNT(COPIES),
    /** 上家の河にある枚数。 */
    KAMICHA_RIVER_COUNT(COPIES),
    /** 今の手が 13 枚相当のとき、その種類が受け入れなら 1。 */
    SELF_UKEIRE(FLAG),
    /** 今の手で形の待ちになる種類なら 1。 */
    SELF_WAIT(FLAG),
    /** 今の手で形の待ちになり、自分の河にもある種類なら 1。 */
    SELF_DISCARDED_WAIT(FLAG),
    /** 下家が立直していて（宣言牌がまだ通っていない立直を含む）、その種類が現物なら 1。 */
    SHIMOCHA_RIICHI_GENBUTSU(FLAG),
    /** 対面が立直していて（宣言牌がまだ通っていない立直を含む）、その種類が現物なら 1。 */
    TOIMEN_RIICHI_GENBUTSU(FLAG),
    /** 上家が立直していて（宣言牌がまだ通っていない立直を含む）、その種類が現物なら 1。 */
    KAMICHA_RIICHI_GENBUTSU(FLAG);

    private final int width;
    private final Set<Architecture> architectures;

    TileInt(int width, Architecture... only) {
      this.width = width;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 牌の種類ごとの実数のフィールド（種類の順に 34 個）。 */
  public enum TileFloat implements NumericField {
    /** 自分の純手牌の枚数 / 4。 */
    SELF_HAND_FRACTION(Tiles.COPIES),
    /** 見えている枚数 / 4。 */
    VISIBLE_FRACTION(Tiles.COPIES),
    /** 自分から見えない枚数 / 4。 */
    UNSEEN_FRACTION(Tiles.COPIES),
    /** その種類をドラにするドラ表示の数。 */
    DORA_MULTIPLICITY(1);

    private final float scale;
    private final Set<Architecture> architectures;

    TileFloat(float scale, Architecture... only) {
      this.scale = scale;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 河のトークンの整数のフィールド（相対席の順に、各席の河の先頭 24 枚）。 */
  public enum RiverInt implements CategoricalField {
    /** 打った席 ＋ 1。 */
    RELATIVE_PLAYER(SEAT),
    /** 河の何枚目か（0 始まり）＋ 1。 */
    INDEX(RIVER),
    /** 牌の種類 ＋ 1。 */
    TILE(ClassicFields.TILE),
    /** 赤なら 1。 */
    IS_AKA(FLAG),
    /** ツモ切りなら 1。 */
    TSUMOGIRI(FLAG),
    /** 立直の宣言牌なら 1。 */
    RIICHI(FLAG),
    /** 鳴かれていれば 1。 */
    CALLED(FLAG),
    /** 打った時の巡目 ＋ 1。 */
    TURN(ClassicFields.TURN),
    /** 局の打牌の通し番号（0 始まり）＋ 1。 */
    GLOBAL_SEQUENCE(SEQUENCE),
    /** 実在する打牌なら 1。 */
    PRESENT(FLAG);

    private final int width;
    private final Set<Architecture> architectures;

    RiverInt(int width, Architecture... only) {
      this.width = width;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 河のトークンの実数のフィールド。 */
  public enum RiverFloat implements NumericField {
    /** 河の何枚目か（0 始まり）/ 24。 */
    PLAYER_PROGRESS(RIVER_SCALE),
    /** 局の打牌の通し番号 / 70。 */
    GLOBAL_PROGRESS(WALL_SCALE);

    private final float scale;
    private final Set<Architecture> architectures;

    RiverFloat(float scale, Architecture... only) {
      this.scale = scale;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 面子のトークンの整数のフィールド（相対席の順に、各席の面子 4 つ）。加槓したポンは加槓の面子として書く。 */
  public enum MeldInt implements CategoricalField {
    /** 面子の持ち主の席 ＋ 1。 */
    RELATIVE_PLAYER(SEAT),
    /** その席で何番目にできた面子か（0 始まり）＋ 1。 */
    INDEX(MELDS),
    /** 面子の種別 ＋ 1（{@link MeldType} の順）。 */
    TYPE(vocabulary(MeldType.class)),
    /** 面子の最小の牌の種類 ＋ 1（チーなら順子の先頭）。 */
    BASE_TILE(ClassicFields.TILE),
    /** 鳴いた牌の種類 ＋ 1。暗槓は 0。 */
    CALLED_TILE(ClassicFields.TILE),
    /** どの向きから鳴いたか ＋ 1（{@link MeldSource} の順）。 */
    SOURCE(vocabulary(MeldSource.class)),
    /** 赤の由来 ＋ 1（{@link MeldAkaSource} の順）。 */
    AKA_SOURCE(vocabulary(MeldAkaSource.class)),
    /** 面子の枚数（3 か 4）。 */
    SIZE(COPIES),
    /** チー・ポン（後で加槓したポンを含む）を鳴いた時点で、持ち主が河に打っていた牌の数 ＋ 1。24 を上限にし、ほかの面子は 0。 */
    CALL_AFTER_RIVER(RIVER),
    /** 槓子になった時点（加槓なら加槓した時点）で、持ち主が河に打っていた牌の数 ＋ 1。24 を上限にし、槓子でなければ 0。 */
    KAN_AFTER_RIVER(RIVER),
    /** 実在する面子なら 1。 */
    PRESENT(FLAG);

    private final int width;
    private final Set<Architecture> architectures;

    MeldInt(int width, Architecture... only) {
      this.width = width;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 面子のトークンの実数のフィールド。 */
  public enum MeldFloat implements NumericField {
    /** チー・ポンを鳴いた時点で、持ち主が河に打っていた牌の数 / 24。ほかの面子は 0。 */
    PLAYER_PROGRESS(RIVER_SCALE),
    /** 槓子になった時点で、持ち主が河に打っていた牌の数 / 24。槓子でなければ 0。 */
    KAN_PROGRESS(RIVER_SCALE);

    private final float scale;
    private final Set<Architecture> architectures;

    MeldFloat(float scale, Architecture... only) {
      this.scale = scale;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 合法手の整数のフィールド（合法手の順）。 */
  public enum ActionInt implements CategoricalField {
    /** 行動の番号 ＋ 1（{@link ClassicVocabulary.ActionBlock} の番号）。 */
    ID(ClassicVocabulary.ACTION_SPACE + 1),
    /** 行動の種別 ＋ 1（{@link ActionType} の順）。 */
    TYPE(vocabulary(ActionType.class)),
    /** 大分類 ＋ 1（{@link ActionGroup} の順）。 */
    GROUP(vocabulary(ActionGroup.class)),
    /** 打つ牌・鳴く牌・槓にする牌の種類 ＋ 1（チーは鳴いた牌）。牌を伴わない行動は 0。 */
    PRIMARY_TILE(TILE),
    /** 物理牌の選び方 ＋ 1（{@link TileSelection} の順）。 */
    TILE_SELECTION(vocabulary(TileSelection.class)),
    /** チーの順子の最小の種類 ＋ 1。チーでなければ 0。 */
    CHI_BASE(TILE),
    /** チーで鳴いた牌が順子の何番目か（0 始まり）＋ 1。チーでなければ 0。 */
    CHI_CALLED_POSITION(CHI_POSITION),
    /** 打つ物理牌の番号 ＋ 1。同じ牌を打つ打牌と立直の打牌は同じ番号になる。打牌でなければ 0。 */
    DISCARD_IDENTITY(ClassicVocabulary.DISCARD_IDENTITIES + 1),
    /** 面子を作る行動なら、できる面子の赤の由来 ＋ 1。ほかの行動は 0。 */
    RESULTING_MELD_AKA_SOURCE(vocabulary(MeldAkaSource.class)),
    /** 行動した後の、自分の立直の状態 ＋ 1。 */
    RESULTING_RIICHI_STATUS(vocabulary(RiichiStatus.class)),
    /** 和了の状況 ＋ 1（{@link WinConditions} の順。和了でない行動は {@code NONE}）。 */
    WIN_CONTEXT(vocabulary(WinConditions.class)),
    /** 立直している和了で、裏ドラを数えるなら 1。 */
    URA_ELIGIBLE(FLAG),
    /** 和了の手なら、和了の後の順位を求めるときに置いた精算の仮定 ＋ 1（{@link SettlementAssumption} の順）。ほかの行動は 0。 */
    SETTLEMENT_ASSUMPTION(vocabulary(SettlementAssumption.class), Architecture.NANO),
    /** 和了の手なら、その和了の後の自分の順位 ＋ 1。ほかの行動は 0。 */
    SOLE_WIN_PROJECTED_RANK(SEAT, Architecture.NANO),
    /** 立直の打牌なら 1。 */
    STARTS_IPPATSU(FLAG),
    /** 面子を作る行動（鳴きと槓）なら 1。 */
    BREAKS_IPPATSU(FLAG),
    /** 行動の後に続く処理 ＋ 1（{@link NextStep} の順）。 */
    FOLLOW_UP_KIND(vocabulary(NextStep.class)),
    /** 局を終える行動（和了と九種九牌）なら 1。 */
    TERMINAL(FLAG);

    private final int width;
    private final Set<Architecture> architectures;

    ActionInt(int width, Architecture... only) {
      this.width = width;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /**
   * 合法手の実数のフィールド（Nano だけ）。和了のフィールドは和了の手だけに書き、ほかの手は 0。和了の点は裏ドラを数えず、ロンの点の
   * 移動には本場を含めない。
   */
  public enum ActionFloat implements NumericField {
    /** 立直の打牌なら供託 1 本の点 / 100000。ほかの手は 0。 */
    RIICHI_DECLARATION_COST(POINTS),
    /** (自分の点 − 立直で払う点) / 100000。 */
    SELF_SCORE_AFTER_DECLARATION(POINTS),
    /** 立直で払った後の (自分の点 − 最高点) / 100000。 */
    SCORE_TO_FIRST_AFTER_DECLARATION(POINTS),
    /** 立直で払った後の (自分の点 − 最低点) / 100000。 */
    SCORE_TO_FOURTH_AFTER_DECLARATION(POINTS),
    /** 和了の翻 / 13（役満は 0）。 */
    NORMALIZED_VISIBLE_HAN_WITHOUT_URA(HAN_SCALE),
    /** 和了の符 / 110（役満は 0）。 */
    NORMALIZED_VISIBLE_FU(FU_SCALE),
    /** 和了の基本点 / 32000。 */
    NORMALIZED_VISIBLE_BASE_POINTS(BASE_SCALE),
    /** 和了による自分の点の移動（受け取りが正）/ 100000。 */
    NORMALIZED_PAYMENT_FLOOR_SELF(POINTS),
    /** 下家の点の移動 / 100000。 */
    NORMALIZED_PAYMENT_FLOOR_SHIMOCHA(POINTS),
    /** 対面の点の移動 / 100000。 */
    NORMALIZED_PAYMENT_FLOOR_TOIMEN(POINTS),
    /** 上家の点の移動 / 100000。 */
    NORMALIZED_PAYMENT_FLOOR_KAMICHA(POINTS),
    /** 和了でめくる裏ドラ表示の数 / 5。 */
    NORMALIZED_URA_INDICATOR_COUNT(WinContext.MAX_INDICATORS),
    /** 和了の後の (自分の点 − 自分の点) / 100000。いつも 0。 */
    NORMALIZED_SOLE_WIN_SCORE_GAP_SELF(POINTS),
    /** 和了の後の (自分の点 − 下家の点) / 100000。 */
    NORMALIZED_SOLE_WIN_SCORE_GAP_SHIMOCHA(POINTS),
    /** 和了の後の (自分の点 − 対面の点) / 100000。 */
    NORMALIZED_SOLE_WIN_SCORE_GAP_TOIMEN(POINTS),
    /** 和了の後の (自分の点 − 上家の点) / 100000。 */
    NORMALIZED_SOLE_WIN_SCORE_GAP_KAMICHA(POINTS);

    private static final Set<Architecture> ARCHITECTURES = ArchitectureScoped.scope(Architecture.NANO);

    private final float scale;

    ActionFloat(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return ARCHITECTURES;
    }
  }

  /**
   * 合法手どうしの位置の対応（合法手の順）。同じ物理牌を打つ手（打牌と立直の打牌）を、方策の計算グラフが互いに参照できるようにする。
   * 埋め込みには入れない。位置は合法手の位置 ＋ 1 で書き、0 は該当なし。打牌でない手はどれも 0。
   */
  public enum ActionRoute implements CategoricalField {
    /** 同じ物理牌を打つ手の代表（打牌があれば打牌、無ければ立直の打牌）なら 1。 */
    DISCARD_IDENTITY_REPRESENTATIVE(FLAG),
    /** 代表の手の位置 ＋ 1。 */
    DISCARD_REPRESENTATIVE_SLOT(SLOTS),
    /** 同じ物理牌を立直せずに打つ手の位置 ＋ 1。 */
    DAMA_SLOT(SLOTS),
    /** 同じ物理牌を立直して打つ手の位置 ＋ 1。 */
    RIICHI_SLOT(SLOTS);

    private final int width;
    private final Set<Architecture> architectures;

    ActionRoute(int width, Architecture... only) {
      this.width = width;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /**
   * 和了の点を決める値（Major だけ）。その場で和了する手ごとと、遷移した後の待ちのロン・ツモごとに書く。点は、Major の点の写像が
   * この値で表を引いて求める。埋め込みには入れない。和了できなければ全て 0。
   */
  public enum WinFact implements CategoricalField {
    /** 役があって和了できれば 1。 */
    VALID(FLAG),
    /** 翻（裏ドラを除く。役満は 0）。 */
    HAN_WITHOUT_URA(HAN),
    /** 符のコード。0 符（役満）は 0、20 符は 1、25 符は 2、30〜110 符は符 / 10 で、110 符を超える符は 11。 */
    FU_CODE(FU_CODES),
    /** 役満の倍率。 */
    YAKUMAN_MULTIPLIER(YAKUMAN),
    /** 包が成り立つなら 1（点の写像はこの和了を使わない）。 */
    REQUIRES_PAO_CORRECTION(FLAG);

    private static final Set<Architecture> ARCHITECTURES = ArchitectureScoped.scope(Architecture.MAJOR);

    private final int width;

    WinFact(int width) {
      this.width = width;
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return ARCHITECTURES;
    }
  }

  /** 合法手を選んだ後の遷移を表す整数のフィールド（合法手ごとに、遷移の順）。 */
  public enum ActionTransitionInt implements CategoricalField {
    /** 遷移の種類 ＋ 1（{@link TransitionKind} の順）。 */
    KIND(vocabulary(TransitionKind.class)),
    /** どの場面で打牌を選ぶか ＋ 1（{@link DiscardContext} の順）。 */
    DISCARD_CONTEXT(vocabulary(DiscardContext.class)),
    /** 打つ牌の行動の番号 ＋ 1（立直の打牌は、同じ牌を打つ打牌の番号）。打たなければ 0。 */
    DISCARD_ACTION_ID(ClassicVocabulary.ACTION_SPACE + 1),
    /** 打つ牌の種類 ＋ 1。打たなければ 0。 */
    DISCARD_TILE(TILE),
    /** 打つ物理牌の選び方 ＋ 1。打たなければ 0。 */
    TILE_SELECTION(vocabulary(TileSelection.class)),
    /** 遷移した後の手で、形の待ちのどれかが自分の河か打つ牌にあれば 1。 */
    RESULTING_DISCARD_FURITEN(FLAG),
    /** 遷移した後も残る、ロンの見逃しによる振聴の種類 ＋ 1（{@link RonFuritenKind} の順）。 */
    RESULTING_RON_FURITEN_KIND(vocabulary(RonFuritenKind.class)),
    /** 遷移した後の手に面子が無く、七対子と国士の形も数えるなら 1。 */
    SPECIAL_HANDS_AVAILABLE(FLAG),
    /** 実在する遷移なら 1。 */
    PRESENT(FLAG);

    private final int width;
    private final Set<Architecture> architectures;

    ActionTransitionInt(int width, Architecture... only) {
      this.width = width;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int width() {
      return width;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /**
   * 遷移した後の手を表す実数のフィールド。受け入れと待ちは、自分の純手牌と面子でまだ 4 枚使い切っていない種類だけを数える。枚数は
   * どれも自分から見えない枚数。役の有無は、立直・一発・嶺上・海底・河底・槍槓・天和・地和を数えずに採点して決める。
   */
  public enum ActionTransitionFloat implements NumericField {
    /** (通常形・七対子・国士のうち最小の向聴 ＋ 1) / 9。 */
    NORMALIZED_MIN_SHANTEN(SHANTEN),
    /** (通常形の向聴 ＋ 1) / 9。 */
    NORMALIZED_NORMAL_SHANTEN(SHANTEN),
    /** (七対子の向聴 ＋ 1) / 9。面子のある手は 0。 */
    NORMALIZED_CHIITOI_SHANTEN(SHANTEN),
    /** (国士の向聴 ＋ 1) / 9。面子のある手は 0。 */
    NORMALIZED_KOKUSHI_SHANTEN(SHANTEN),
    /** 受け入れの枚数 / 136。 */
    NORMALIZED_UKEIRE_COUNT(LIVE),
    /** 残り枚数のある受け入れの種類の数 / 34。 */
    NORMALIZED_UKEIRE_KINDS(KIND_SCALE),
    /** 形の待ちの種類の数 / 34。 */
    NORMALIZED_SHAPE_WAIT_KINDS(KIND_SCALE),
    /**
     * 次の 1 回のツモで受け入れを 1 枚以上引く確率。受け入れの枚数を k、見えない枚数を n として 1 − Π_{i<1} (n − k − i) / (n − i)。
     */
    CURRENT_UKEIRE_HIT_WITHIN_1_SELF_DRAW(1),
    /** 2 回のツモまでに受け入れを引く確率。 */
    CURRENT_UKEIRE_HIT_WITHIN_2_SELF_DRAWS(1),
    /** 3 回のツモまでに受け入れを引く確率。 */
    CURRENT_UKEIRE_HIT_WITHIN_3_SELF_DRAWS(1),
    /** 形の待ちの枚数 / 136。 */
    NORMALIZED_LIVE_SHAPE_WAIT_COPIES(LIVE),
    /** 立直を数えなくてもロンで役のある待ちの種類の数 / 34。 */
    NORMALIZED_INTRINSIC_RON_YAKU_WAIT_KINDS(KIND_SCALE),
    /** 立直を数えなくてもロンで役のある待ちの枚数 / 136。 */
    NORMALIZED_INTRINSIC_RON_YAKU_WAIT_COPIES(LIVE),
    /** 立直を数えなくてもツモで役のある待ちの種類の数 / 34。 */
    NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_KINDS(KIND_SCALE),
    /** 立直を数えなくてもツモで役のある待ちの枚数 / 136。 */
    NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_COPIES(LIVE),
    /** 1 向聴の手で、引いて 1 枚打てば、振聴でなくロンで役のある聴牌になる種類の数 / 34。 */
    NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_KINDS(KIND_SCALE),
    /** 上の種類の枚数 / 136。 */
    NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_COPIES(LIVE),
    /** 1 向聴の手で、引いてもロンで役のある聴牌には振聴でしかならない種類の数 / 34。 */
    NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_KINDS(KIND_SCALE),
    /** 上の種類の枚数 / 136。 */
    NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_COPIES(LIVE),
    /** 1 向聴の手で、引いて 1 枚打てばツモで役のある聴牌になる種類の数 / 34。 */
    NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_KINDS(KIND_SCALE),
    /** 上の種類の枚数 / 136。 */
    NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_COPIES(LIVE),
    /** 純手牌と面子のドラの数（牌ごとに、その牌をドラにするドラ表示の数を掛ける）/ 20。 */
    NORMALIZED_DORA_COUNT(20),
    /** 純手牌と面子の赤の枚数 / 3。 */
    NORMALIZED_AKA_TILE_COUNT(Tiles.NUMBER_SUITS),
    /** 打つ牌の種類の見えない枚数 / 4。打たなければ 0。 */
    NORMALIZED_PRIMARY_TILE_UNSEEN_COPIES(Tiles.COPIES),
    /** 門前を崩す面子が無ければ 1。 */
    MENZEN(1),
    /** 面子の数 / 4。 */
    NORMALIZED_MELD_COUNT(ClassicSchema.MELD_TOKENS),
    /** 純手牌の枚数 / 14。 */
    NORMALIZED_CONCEALED_TILE_COUNT(Variant.HAND + 1);

    private final float scale;
    private final Set<Architecture> architectures;

    ActionTransitionFloat(float scale, Architecture... only) {
      this.scale = scale;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /** 遷移した後の待ちごとの点を表す実数のフィールド（Nano だけ。待ちの種類の昇順に 13 まで）。役が無ければ 0。和了牌は赤でない牌として数え、裏ドラは数えない。 */
  public enum ActionTransitionWaitFloat implements NumericField {
    /** ロンの翻 / 13（役満は 0）。 */
    RON_NORMALIZED_VISIBLE_HAN(HAN_SCALE),
    /** ロンの符 / 110（役満は 0）。 */
    RON_NORMALIZED_FU(FU_SCALE),
    /** ロンの基本点 / 32000。 */
    RON_NORMALIZED_BASE_POINTS(BASE_SCALE),
    /** ツモの翻 / 13（役満は 0）。 */
    TSUMO_NORMALIZED_VISIBLE_HAN(HAN_SCALE),
    /** ツモの符 / 110（役満は 0）。 */
    TSUMO_NORMALIZED_FU(FU_SCALE),
    /** ツモの基本点 / 32000。 */
    TSUMO_NORMALIZED_BASE_POINTS(BASE_SCALE);

    private static final Set<Architecture> ARCHITECTURES = ArchitectureScoped.scope(Architecture.NANO);

    private final float scale;

    ActionTransitionWaitFloat(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return ARCHITECTURES;
    }
  }

  /**
   * 局が始まった時点の値（1 行に 1 つずつ）。価値の計算だけが読み、方策は読まない。GRP の事前分布が与えられたときは、局の始まりに
   * 公開されている値と、値があることを示す印を書く。与えられなければ分布を一様にし、ほかは 0 にする。
   */
  public enum BoundaryFloat implements NumericField {
    /** GRP の事前分布。絶対席 × 順位の周辺確率を、席の順に 4 つずつ並べる。 */
    GRP_RANK(ClassicSchema.SEATS * ClassicSchema.SEATS, 1),
    /** その局の通し番号 / 8。 */
    KYOKU_INDEX(1, 8),
    /** 局の始まりの本場の数 / 8。 */
    HONBA(1, 8),
    /** 局の始まりの供託の本数 / 4。 */
    KYOTAKU(1, 4),
    /** 局の始まりの点（絶対席の順）/ 10000。 */
    SCORE(ClassicSchema.SEATS, 10_000),
    /** 局の始まりの値と GRP の事前分布があれば 1。 */
    PRESENT(1, 1);

    private final int count;
    private final float scale;
    private final Set<Architecture> architectures;

    BoundaryFloat(int count, float scale, Architecture... only) {
      this.count = count;
      this.scale = scale;
      architectures = ArchitectureScoped.scope(only);
    }

    @Override
    public int count() {
      return count;
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public Set<Architecture> architectures() {
      return architectures;
    }
  }

  /**
   * 遷移の牌のコード（牌の種類ごとに 1 つ、34 個）に入れるフラグ。コード = (純手牌の枚数 × 5 ＋ 受け入れの残り) × 64 ＋ フラグの bit の和 ＋ 1
   * で、0 はパディング。受け入れの残りは、その種類が受け入れなら見えない枚数、ほかは 0。フラグの bit は宣言の順に 32、16、8、4、2、1。
   */
  public enum TileFlag {
    /** 形の待ち。 */
    SHAPE_WAIT,
    /** ロンで役のある待ち。 */
    RON_YAKU_WAIT,
    /** ツモで役のある待ち。 */
    TSUMO_YAKU_WAIT,
    /** 遷移で打つ牌の種類。 */
    DISCARDED,
    /** 合法手で作る面子の牌の種類（チー・ポンの後の打牌では、鳴いた面子の牌の種類）。 */
    CALLED_INTO_MELD,
    /** 待ちの 5 で、赤 5 がまだ見えていない。 */
    AKA_WINNING_TILE;

    private static final int FLAGS = values().length;

    /** コードの値の数（パディングを含む）。 */
    public static final int CODES = COPIES * COPIES * (1 << FLAGS) + 1;

    public int bit() {
      return 1 << (FLAGS - 1 - ordinal());
    }

    /**
     * 牌の種類 1 つ分のコードを返す。
     *
     * @param concealed 遷移した後の純手牌の枚数
     * @param ukeire 受け入れの残り
     * @param flags フラグの bit の和
     * @return 1 以上 {@link #CODES} 未満のコード
     */
    public static int code(int concealed, int ukeire, int flags) {
      return (concealed * COPIES + ukeire) * (1 << FLAGS) + flags + 1;
    }
  }
}
