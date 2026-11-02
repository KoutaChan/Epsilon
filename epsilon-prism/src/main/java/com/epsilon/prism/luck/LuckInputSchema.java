package com.epsilon.prism.luck;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.GameState;
import com.epsilon.core.RoundLimits;
import com.epsilon.core.Tile;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.SchemaDigest;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.Ids;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.BoundedField;
import com.epsilon.prism.input.InputSchema.Wind;
import com.epsilon.prism.input.TokenLayout;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * 運の網 Ω の入力の欄。トークンは局 1・席 4・牌 34 の 39 個で、出来事のトークンはない。
 *
 * <p>作りは {@link InputSchema} と同じ: 欄はトークンの型ごとの enum に置き、定数ごとに書き方・語彙・繰り返し（{@link FieldSpec}）を持たせ、列と
 * 埋め込みの名前空間は {@link TokenLayout} が enum から一度だけ作る。値の決まりも同じで、0 は常に「なし・0 個・偽」、enum は ordinal + 1、牌・席・
 * 0 始まりの番号は番号 + 1（{@link Ids}）。打つ網の欄とは読む情報が違う（全員の手牌とプールを見る）ので、Ω だけの
 * enum にする。
 *
 * <p>席は行為者から見た相対席（0 行為者、1 下家、2 対面、3 上家）、牌は色を正準化した順（{@link LuckSuits}）。点と支払いは整数から作った区分
 * （{@link WinGrade}）で渡し、BF16 の数値にしない。出来事の結果（O° の結果の欄）は入力に入れない。
 *
 * <p>{@link #fingerprint()} は、読む O° の指紋、欄の名前・順・書き方・語彙・繰り返し、値の enum、数値の欄の正規化、トークンの並び、関係の符号と
 * 色の正準化の定義から作る。版の番号は持たない。
 */
public final class LuckInputSchema {

  /** 局の欄の配置。 */
  public static final TokenLayout<RoundField> ROUND = TokenLayout.of(RoundField.class);

  /** 席の欄の配置。 */
  public static final TokenLayout<SeatField> SEAT = TokenLayout.of(SeatField.class);

  /** 牌の欄の配置。 */
  public static final TokenLayout<TileField> TILE = TokenLayout.of(TileField.class);

  /** 生きた枚数の欄を割る数。 */
  private static final float LIVE_TILE_SCALE = 16f;

  private LuckInputSchema() {}

  /**
   * 入力の指紋を返す。Ω の網の model.json に書き、開くときに照らす。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /** トークンの型。行の中の並びは宣言の順（局 1・席 4・牌 34）。 */
  public enum Token {
    /** 局（行に 1 個）。 */
    ROUND(1),
    /** 席（行為者から見た相対席の順に 4 個）。 */
    SEAT(GameState.NUM_PLAYERS),
    /** 牌（正準の色の順に 34 個）。 */
    TILE(Tile.NUM_TILE_TYPES);

    /** 行のトークンの数。 */
    public static final int ROW_TOKENS;

    private static final int[] FIRST;

    static {
      Token[] tokens = values();
      FIRST = new int[tokens.length];
      int position = 0;
      for (Token token : tokens) {
        FIRST[token.ordinal()] = position;
        position += token.perRow;
      }
      ROW_TOKENS = position;
    }

    private final int perRow;

    Token(int perRow) {
      this.perRow = perRow;
    }

    /** 行あたりの個数。 */
    public int perRow() {
      return perRow;
    }

    /** 行の中の最初の位置。 */
    public int first() {
      return FIRST[ordinal()];
    }

    /** 欄の配置。 */
    public TokenLayout<?> layout() {
      return switch (this) {
        case ROUND -> LuckInputSchema.ROUND;
        case SEAT -> LuckInputSchema.SEAT;
        case TILE -> LuckInputSchema.TILE;
      };
    }

    /** 数値の欄の数。 */
    public int scalars() {
      return switch (this) {
        case ROUND -> RoundScalar.values().length;
        case SEAT -> SeatScalar.values().length;
        case TILE -> 0;
      };
    }
  }

  /**
   * 和了の格（基本点の段。裏ドラは数えず、条件は自風・場風・立直だけ。一発・海底・嶺上などは数えない）。役がなければ欄は 0（なし）。
   */
  public enum WinGrade {
    /** 基本点 960 未満。 */
    LOW(0),
    /** 960 以上、満貫未満。 */
    MIDDLE(960),
    MANGAN(2000),
    HANEMAN(3000),
    BAIMAN(4000),
    SANBAIMAN(6000),
    /** 役満（倍の役満を含む）。 */
    YAKUMAN(8000);

    private static final WinGrade[] VALUES = values();

    private final int minimum;

    WinGrade(int minimum) {
      this.minimum = minimum;
    }

    /** この段の最小の基本点。 */
    public int minimum() {
      return minimum;
    }

    /**
     * 基本点の段を返す。
     *
     * @param basePoints 基本点（正）
     * @return 段
     * @throws IllegalArgumentException 基本点が正でないとき
     */
    public static WinGrade of(int basePoints) {
      if (basePoints <= 0) {
        throw new IllegalArgumentException("base points must be positive: " + basePoints);
      }
      WinGrade grade = LOW;
      for (WinGrade candidate : VALUES) {
        if (basePoints >= candidate.minimum) {
          grade = candidate;
        }
      }
      return grade;
    }
  }

  /** 局のトークン（行に 1 個）の欄。 */
  public enum RoundField implements BoundedField {
    EVENT(FieldSpec.of(LuckEvent.class)),
    ROUND_WIND(FieldSpec.of(Wind.class)),
    /** 局（東 1 局 = 1 … 西 4 局 = 12）。 */
    KYOKU(FieldSpec.ids(HanchanProgression.WEST_END_KYOKU + 1)),
    /** 本場（{@link InputSchema#STICK_SATURATION} 以上はまとめる）。 */
    HONBA(FieldSpec.count(InputSchema.STICK_SATURATION)),
    /** 供託の本数（同上）。 */
    KYOTAKU(FieldSpec.count(InputSchema.STICK_SATURATION)),
    /** 成立した槓の数（4 席の和）。 */
    KANS(FieldSpec.count(RoundLimits.MAX_KANS)),
    /** 出来事の前に開いているドラ表示の数（初めの表示を含む）。 */
    INDICATORS(FieldSpec.count(Limits.MAX_INDICATORS)),
    /** 嶺上待ち（槓をして、その嶺上のツモがまだ）。 */
    RINSHAN_PENDING(FieldSpec.flag());

    private final FieldSpec spec;

    RoundField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 席のトークン（行為者から見た相対席の順に 4 個）の欄。 */
  public enum SeatField implements BoundedField {
    /** 行為者から見た相対席（行為者 = 1、下家 = 2、対面 = 3、上家 = 4）。 */
    RELATIVE_SEAT(FieldSpec.ids(GameState.NUM_PLAYERS)),
    /** 自風（東が親）。 */
    WIND(FieldSpec.of(Wind.class)),
    /** 今の順位（同点は ScoreRanking の席順）。 */
    RANK(FieldSpec.ids(GameState.NUM_PLAYERS)),
    RIICHI(FieldSpec.of(RiichiState.class)),
    IPPATSU(FieldSpec.flag()),
    /** 見逃しの振聴（同巡の振聴と立直後の振聴）。 */
    FURITEN(FieldSpec.flag()),
    /** 待ちが自分の河にある（捨て牌の振聴）。 */
    DISCARD_FURITEN(FieldSpec.flag()),
    MENZEN(FieldSpec.flag()),
    /** まだ打牌していない（第一ツモの旗）。 */
    FIRST_DRAW(FieldSpec.flag()),
    /** 副露の数（暗槓を含む）。 */
    MELDS(FieldSpec.count(RoundLimits.MAX_MELDS)),
    KANS(FieldSpec.count(RoundLimits.MAX_KANS)),
    /** 河の打牌の数（鳴かれた打牌を含む）。 */
    DISCARDS(FieldSpec.count(RoundLimits.MAX_DISCARDS)),
    /** 手（副露を除く）の最小の向聴（聴牌は 0。行為者は出来事の前の手で数える）。 */
    SHANTEN(FieldSpec.ids(Limits.MAX_SHANTEN + 1)),
    /** 待ちの種類の数（4 枚とも自分が持つ牌は待ちにしない）。 */
    WAITS(FieldSpec.count(Limits.MAX_WAIT_KINDS)),
    /** 手と副露のドラ（表ドラと赤、裏なし）。 */
    DORA(FieldSpec.count(Limits.MAX_DORA));

    private final FieldSpec spec;

    SeatField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 牌のトークン（正準の色の順に 34 個）の欄。 */
  public enum TileField implements BoundedField {
    /** 牌の種類（正準）。 */
    TILE(FieldSpec.ids(Tile.NUM_TILE_TYPES)),
    /** 相対席ごとの手（副露を除く）の枚数。 */
    HELD(FieldSpec.count(Tile.TILES_PER_TYPE).times(GameState.NUM_PLAYERS)),
    /** 相対席ごとに、手にこの種の赤がある。 */
    HELD_AKA(FieldSpec.flag().times(GameState.NUM_PLAYERS)),
    /** 公開で見える枚数（鳴かれていない河・副露・ドラ表示）。 */
    VISIBLE(FieldSpec.count(Tile.TILES_PER_TYPE)),
    /** プールの枚数（4 − 公開 − 全員の手。結果はこの上で一様に出る）。 */
    POOL(FieldSpec.count(Tile.TILES_PER_TYPE)),
    /** この種の赤がプールにある。 */
    POOL_AKA(FieldSpec.flag()),
    /** ドラの倍率（この牌を指す表示の数）。 */
    DORA(FieldSpec.count(Tile.TILES_PER_TYPE)),
    /**
     * この牌が槓ドラの表示になったときにドラになる牌の、相対席ごとの所持（手と副露）の枚数。槓ドラの出来事の結果ごとの得点の動きを、牌の
     * トークンから直に読むために置く。
     */
    DORA_HELD(FieldSpec.count(Tile.TILES_PER_TYPE).times(GameState.NUM_PLAYERS)),
    /** 相対席ごとに、この牌が河にある（鳴かれた打牌を含む）。 */
    RIVER(FieldSpec.flag().times(GameState.NUM_PLAYERS)),
    /** 相対席ごとに、この牌が待ち（席の欄の WAITS と同じ数え方）。 */
    WAIT(FieldSpec.flag().times(GameState.NUM_PLAYERS)),
    /** 行為者がこの牌で向聴を下げる（行為者の受け入れ）。 */
    UKEIRE(FieldSpec.flag()),
    /** 行為者がこの牌（赤でない）をツモって和了したときの格（プールにない牌は 0）。 */
    TSUMO(FieldSpec.of(WinGrade.class)),
    /** 行為者がこの種の赤をツモって和了したときの格（赤がプールにある 5 だけ）。 */
    TSUMO_AKA(FieldSpec.of(WinGrade.class)),
    /** 下家・対面・上家がこの牌でロンしたときの格（振聴・役なしは 0）。 */
    RON(FieldSpec.of(WinGrade.class).times(GameState.NUM_PLAYERS - 1));

    private final FieldSpec spec;

    TileField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 数値の欄（FP32 で送り、網が計算の型に丸める）。値は scale で割ってから送る。 */
  public sealed interface Scalar permits RoundScalar, SeatScalar {

    /**
     * 正規化の定数を返す。
     *
     * @return 値を割る数
     */
    float scale();
  }

  /** 局の数値の欄。 */
  public enum RoundScalar implements Scalar {
    /** 王牌を除く山の残り（出来事の前）。 */
    WALL(RoundLimits.MAX_WALL),
    /** プールの枚数の和 N。 */
    POOL(Tile.NUM_TILES);

    private final float scale;

    RoundScalar(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /** 席の数値の欄。 */
  public enum SeatScalar implements Scalar {
    SCORE(100_000f),
    /** 行為者との点差（この席 − 行為者）。 */
    GAP_TO_ACTOR(10_000f),
    /** 待ちのプールの枚数の和（生きた枚数）。 */
    LIVE_WAITS(LIVE_TILE_SCALE);

    private final float scale;

    SeatScalar(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /** 遅延で作る（初期化の順の罠を避ける）。 */
  private static final class Fingerprint {

    private static final String VALUE = digest().hex();

    private static SchemaDigest digest() {
      SchemaDigest digest =
          SchemaDigest.named("epsilon-prism-v2-luck-input")
              .with("observation", LuckSchema.fingerprint());
      Set<Class<? extends Enum<?>>> valueTypes = new LinkedHashSet<>();
      for (Class<?> type : List.of(RoundField.class, SeatField.class, TileField.class)) {
        StringBuilder fields = new StringBuilder();
        for (Object constant : type.getEnumConstants()) {
          FieldSpec spec = ((BoundedField) constant).spec();
          fields.append(constant).append(':').append(spec.coding()).append(':');
          fields.append(spec.vocabulary()).append('x').append(spec.repeats()).append(',');
          if (spec.values() != null) {
            valueTypes.add(spec.values());
          }
        }
        digest.with(SchemaDigest.name(type), fields);
      }
      for (Class<? extends Enum<?>> type : valueTypes) {
        digest.values(type);
      }
      StringBuilder grades = new StringBuilder();
      for (WinGrade grade : WinGrade.values()) {
        grades.append(grade).append('=').append(grade.minimum()).append(',');
      }
      for (Class<?> type : SchemaDigest.sorted(Scalar.class.getPermittedSubclasses())) {
        StringBuilder scalars = new StringBuilder();
        for (Object constant : type.getEnumConstants()) {
          scalars.append(constant).append('/').append(((Scalar) constant).scale()).append(',');
        }
        digest.with(SchemaDigest.name(type), scalars);
      }
      StringBuilder tokens = new StringBuilder();
      for (Token token : Token.values()) {
        tokens.append(token).append('x').append(token.perRow()).append(',');
      }
      return digest
          .with(SchemaDigest.name(WinGrade.class), grades)
          .with(SchemaDigest.name(Token.class), tokens)
          .with("relation", LuckRelation.DEFINITION)
          .with("suits", LuckSuits.DEFINITION)
          .with("outcomes", LuckSchema.OUTCOMES);
    }
  }
}
