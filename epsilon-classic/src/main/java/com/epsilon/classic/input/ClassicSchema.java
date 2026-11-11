package com.epsilon.classic.input;

import com.epsilon.agent.observe.codec.SchemaDigest;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicFields.ActionFloat;
import com.epsilon.classic.input.ClassicFields.ActionInt;
import com.epsilon.classic.input.ClassicFields.ActionRoute;
import com.epsilon.classic.input.ClassicFields.ActionTransitionFloat;
import com.epsilon.classic.input.ClassicFields.ActionTransitionInt;
import com.epsilon.classic.input.ClassicFields.ActionTransitionWaitFloat;
import com.epsilon.classic.input.ClassicFields.BoundaryFloat;
import com.epsilon.classic.input.ClassicFields.MeldFloat;
import com.epsilon.classic.input.ClassicFields.MeldInt;
import com.epsilon.classic.input.ClassicFields.PlayerFloat;
import com.epsilon.classic.input.ClassicFields.PlayerInt;
import com.epsilon.classic.input.ClassicFields.RiverFloat;
import com.epsilon.classic.input.ClassicFields.RiverInt;
import com.epsilon.classic.input.ClassicFields.RoundFloat;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicFields.TileFlag;
import com.epsilon.classic.input.ClassicFields.TileFloat;
import com.epsilon.classic.input.ClassicFields.TileInt;
import com.epsilon.classic.input.ClassicFields.WinFact;
import com.epsilon.classic.input.ClassicVocabulary.ActionBlock;
import com.epsilon.classic.input.ClassicVocabulary.WaitWinType;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

/**
 * アーキテクチャ（{@link Architecture}）ごとに、入力の各要素をどこに置くかを決める。フィールド・トークン・テンソルを宣言した順に
 * 並べて、各要素のオフセット、1 行の要素数、埋め込みの名前空間の先頭を最初に一度だけ求める。不変なので、スレッド間で共有できる。
 *
 * <p>状態の行（{@link Tensor#STATE_CATEGORIES}・{@link Tensor#STATE_NUMERICS}）には、{@link StateToken} の種類を宣言した順に
 * 置く。トークンの種類ごとに、フィールドの並びをトークンの数だけ繰り返す。合法手の行では合法手ごとに、遷移の行では合法手 × 遷移ごとに、
 * フィールドの並びを繰り返す。待ちの行には、遷移ごとに待ちの種類を昇順に最大 13 個並べる。
 *
 * <p>埋め込む整数のフィールドは、埋め込みの表の中にそれぞれ自分の名前空間（幅の数だけの行）を持つ。名前空間の先頭は、フィールドの
 * 幅を宣言の順に足し上げて決める。状態の表では局・席・牌・河・面子のフィールドがこの順に続き、合法手の表では合法手のフィールドの
 * 後に遷移のフィールドが続く。同じフィールドは、どのトークンの位置にあっても同じ名前空間を引く。
 */
public final class ClassicSchema {

  /** 席の数（四人打ちだけ）。 */
  public static final int SEATS = 4;

  /** 1 席の河のトークン数。これより後の打牌はトークンにせず、打牌を数えるフィールドもこの数を上限にする。 */
  public static final int RIVER_TOKENS = 24;

  /** 1 席の面子のトークン数。 */
  public static final int MELD_TOKENS = 4;

  /** 1 行に置く合法手の数の上限。 */
  public static final int MAX_LEGAL = 40;

  /** 1 つの合法手に置く遷移の数の上限。 */
  public static final int MAX_TRANSITIONS = 16;

  /** 1 つの遷移に置く待ちの種類の上限（国士の 13 面待ち）。 */
  public static final int MAX_WAITS = 13;

  /** 待ちの役のビットを、1 つの値に何ビットずつまとめるか。 */
  public static final int YAKU_CHUNK_BITS = 14;

  /** 役のビットをまとめた値の数（ロンとツモそれぞれで）。 */
  public static final int YAKU_CHUNKS =
      (ClassicVocabulary.Yaku.values().length + YAKU_CHUNK_BITS - 1) / YAKU_CHUNK_BITS;

  /** 待ちの役をまとめた値の幅（14 ビットで表せる値の数 ＋ 1。0 はパディング）。 */
  public static final int YAKU_CHUNK_VALUES = (1 << YAKU_CHUNK_BITS) + 1;

  /** 待ちの種類の値の幅（種類の数 ＋ 1。0 はパディング）。 */
  public static final int WAIT_TILE_VALUES = Tiles.KINDS + 1;

  /** 公開の履歴で、クエリの席（相対席）1 つに置くトークンの数（河と面子のトークンを合わせた数）。 */
  public static final int HISTORY_TOKENS = SEATS * (RIVER_TOKENS + MELD_TOKENS);

  /** 公開の履歴で、1 トークンが使う要素数。 */
  public static final int HISTORY_WORDS = 2;

  /** 公開の履歴を引く表で、時点の前後を表すフィールド 1 つが使う行数（関係の種類の数）。 */
  public static final int HISTORY_SEGMENT_ROWS = PublicHistory.Relation.values().length;

  /**
   * 公開の履歴を引く表の行数。1 トークンの値は 4 ビットずつのフィールドに分かれていて、フィールドごとに宣言の順で行を続ける。時点の
   * 前後を表すフィールドは {@link #HISTORY_SEGMENT_ROWS} 行ずつ、最後のフィールド（トークンの席）は席の数だけ行を使う。
   */
  public static final int HISTORY_TABLE_ROWS =
      (PublicHistory.Nibble.values().length - 1) * HISTORY_SEGMENT_ROWS + SEATS;

  /** 状態のトークンの種類。宣言の順に、状態の行へ続けて置く。 */
  public enum StateToken {
    /** 局（1 つ）。 */
    ROUND(RoundInt.class, RoundFloat.class, 1),
    /** 席（相対席の順に 4 つ）。 */
    PLAYER(PlayerInt.class, PlayerFloat.class, SEATS),
    /** 牌の種類（種類の順に 34）。 */
    TILE(TileInt.class, TileFloat.class, Tiles.KINDS),
    /** 河（相対席の順に、各席の河の先頭 24 枚）。 */
    RIVER(RiverInt.class, RiverFloat.class, SEATS * RIVER_TOKENS),
    /** 面子（相対席の順に、各席で成立した順に 4 つ）。 */
    MELD(MeldInt.class, MeldFloat.class, SEATS * MELD_TOKENS);

    private final Class<? extends Enum<?>> categorical;
    private final Class<? extends Enum<?>> numeric;
    private final int count;

    StateToken(Class<? extends Enum<?>> categorical, Class<? extends Enum<?>> numeric, int count) {
      this.categorical = categorical;
      this.numeric = numeric;
      this.count = count;
    }

    /** トークンの数。 */
    public int count() {
      return count;
    }

    /** 整数のフィールドの enum。 */
    public Class<? extends Enum<?>> categorical() {
      return categorical;
    }

    /** 実数のフィールドの enum。 */
    public Class<? extends Enum<?>> numeric() {
      return numeric;
    }
  }

  private static final Map<Architecture, ClassicSchema> BY_ARCHITECTURE = schemas();

  private final Architecture architecture;
  private final Map<Class<?>, FieldLayout<?>> layouts = new LinkedHashMap<>();
  private final int[] widths = new int[Tensor.values().length];
  private final int[] categoricalStarts = new int[StateToken.values().length];
  private final int[] numericStarts = new int[StateToken.values().length];
  private final int historyStart;
  private final int[] stateOffsets;
  private final int[] actionOffsets;
  private final int[] transitionOffsets;
  private final int stateDictionary;
  private final int actionDictionary;
  private final String fingerprint;

  private ClassicSchema(Architecture architecture) {
    this.architecture = architecture;
    for (Class<? extends Enum<?>> type : fieldTypes()) {
      layouts.put(type, newLayout(type));
    }
    int categorical = 0;
    int numeric = 0;
    for (StateToken token : StateToken.values()) {
      categoricalStarts[token.ordinal()] = categorical;
      numericStarts[token.ordinal()] = numeric;
      categorical += token.count * stride(token.categorical);
      numeric += token.count * stride(token.numeric);
    }
    historyStart = architecture.publicHistory() ? categorical : -1;
    int history = historyStart < 0 ? 0 : SEATS * HISTORY_TOKENS * HISTORY_WORDS;
    put(Tensor.STATE_CATEGORIES, categorical + history);
    put(Tensor.ACTION_CATEGORIES, stride(ActionInt.class));
    put(Tensor.ACTION_ROUTES, stride(ActionRoute.class));
    put(Tensor.POINT_LEDGER_100, SEATS);
    put(Tensor.ACTION_WIN_FACTS, stride(WinFact.class));
    put(Tensor.TRANSITION_CATEGORIES, stride(ActionTransitionInt.class));
    put(Tensor.TRANSITION_TILES, Tiles.KINDS);
    put(Tensor.WAIT_TILE_IDS, MAX_WAITS);
    put(Tensor.WAIT_YAKUS, MAX_WAITS * WaitWinType.values().length * YAKU_CHUNKS);
    put(Tensor.WAIT_WIN_FACTS, MAX_WAITS * WaitWinType.values().length * stride(WinFact.class));
    put(Tensor.STATE_NUMERICS, numeric);
    put(Tensor.BOUNDARY_CONTEXT, stride(BoundaryFloat.class));
    put(Tensor.ACTION_NUMERICS, stride(ActionFloat.class));
    put(Tensor.TRANSITION_NUMERICS, stride(ActionTransitionFloat.class));
    put(Tensor.WAIT_SCORES, MAX_WAITS * stride(ActionTransitionWaitFloat.class));
    stateOffsets = new int[categorical];
    int state = 0;
    for (StateToken token : StateToken.values()) {
      state = namespaces(token.categorical, token.count, state, categoricalStarts[token.ordinal()], stateOffsets);
    }
    stateDictionary = state;
    actionOffsets = new int[stride(ActionInt.class)];
    transitionOffsets = new int[stride(ActionTransitionInt.class)];
    int action = namespaces(ActionInt.class, 1, 0, 0, actionOffsets);
    actionDictionary = namespaces(ActionTransitionInt.class, 1, action, 0, transitionOffsets);
    fingerprint = describe().hex();
  }

  public static ClassicSchema of(Architecture architecture) {
    return BY_ARCHITECTURE.get(architecture);
  }

  public Architecture architecture() {
    return architecture;
  }

  /** 入力の配置のフィンガープリント（SHA-256 の先頭 8 B を 16 進で書いた文字列）。チェックポイントのヘッダーに書き、開くときに照合する。 */
  public String fingerprint() {
    return fingerprint;
  }

  /**
   * フィールドの enum の配置を返す。
   *
   * @param type フィールドの enum の型
   * @throws IllegalArgumentException 入力のフィールドの enum でないとき
   */
  @SuppressWarnings("unchecked")
  public <E extends Enum<E> & InputField> FieldLayout<E> layout(Class<E> type) {
    return (FieldLayout<E>) layoutOf(type);
  }

  /** フィールドの enum の配置（型引数を付けずに返す）。入力のフィールドの enum でなければ IllegalArgumentException を投げる。 */
  FieldLayout<?> layoutOf(Class<?> type) {
    FieldLayout<?> layout = layouts.get(type);
    if (layout == null) {
      throw new IllegalArgumentException(type.getSimpleName() + " is not an input field");
    }
    return layout;
  }

  /** テンソルが 1 単位（行・合法手・遷移）あたりに持つ要素数。このアーキテクチャに無いテンソルは 0。 */
  public int width(Tensor tensor) {
    return tensor.in(architecture) ? widths[tensor.ordinal()] : 0;
  }

  /** テンソルの 1 行の要素数。合法手と遷移の数は、区分の上限で数える。 */
  public int rowWidth(Tensor tensor, Bucket bucket) {
    int units =
        switch (tensor.axis()) {
          case ROW -> 1;
          case ACTION -> bucket.actions();
          case TRANSITION -> bucket.actions() * bucket.transitions();
        };
    return width(tensor) * units;
  }

  /** 状態の整数の行か実数の行の中で、そのトークンの種類が始まる要素の位置。 */
  public int stateStart(StateToken token, Tensor.Slab slab) {
    return slab == Tensor.Slab.CATEGORICAL ? categoricalStarts[token.ordinal()] : numericStarts[token.ordinal()];
  }

  /** 状態の整数の行の中で、公開の履歴が始まる要素の位置。公開の履歴を持たないアーキテクチャでは −1。 */
  public int historyStart() {
    return historyStart;
  }

  /** 状態の整数の行のうち公開の履歴より前の要素について、要素ごとに埋め込みの名前空間の先頭を返す。呼ぶたびに新しい配列を返す。 */
  public int[] stateOffsets() {
    return stateOffsets.clone();
  }

  /** 合法手の整数のフィールドの要素ごとに、合法手の名前空間の先頭を返す。呼ぶたびに新しい配列を返す。 */
  public int[] actionOffsets() {
    return actionOffsets.clone();
  }

  /** 遷移の整数のフィールドの要素ごとに、合法手の名前空間の先頭を返す（合法手のフィールドの後に続く）。呼ぶたびに新しい配列を返す。 */
  public int[] transitionOffsets() {
    return transitionOffsets.clone();
  }

  /** 状態の埋め込みの表の行数。 */
  public int stateDictionary() {
    return stateDictionary;
  }

  /** 合法手と遷移の埋め込みの表の行数。 */
  public int actionDictionary() {
    return actionDictionary;
  }

  /** 宣言された全てのフィールドの enum。状態のトークン、合法手、遷移、局の始まりの値の順に並べる。 */
  public static List<Class<? extends Enum<?>>> fieldTypes() {
    List<Class<? extends Enum<?>>> types = new ArrayList<>();
    for (StateToken token : StateToken.values()) {
      types.add(token.categorical);
      types.add(token.numeric);
    }
    types.addAll(
        List.<Class<? extends Enum<?>>>of(
            ActionInt.class,
            ActionFloat.class,
            ActionRoute.class,
            WinFact.class,
            ActionTransitionInt.class,
            ActionTransitionFloat.class,
            ActionTransitionWaitFloat.class,
            BoundaryFloat.class));
    return types;
  }

  private void put(Tensor tensor, int width) {
    widths[tensor.ordinal()] = width;
  }

  private int stride(Class<?> type) {
    return layouts.get(type).stride();
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private FieldLayout<?> newLayout(Class<? extends Enum<?>> type) {
    return new FieldLayout(type, architecture);
  }

  /**
   * 整数のフィールドの enum に、幅を足し上げて名前空間を割り当てる。各トークンの要素ごとに、名前空間の先頭を書き込む。
   *
   * @param type 整数のフィールドの enum
   * @param count トークンの数
   * @param cursor 最初のフィールドの名前空間の先頭
   * @param start 書き込み先で最初に書く位置
   * @param out 書き込み先
   * @return 最後のフィールドの名前空間の終わり（次の名前空間の先頭）
   */
  private int namespaces(Class<?> type, int count, int cursor, int start, int[] out) {
    FieldLayout<?> layout = layouts.get(type);
    int[] firsts = new int[layout.stride()];
    for (Object field : layout.fields()) {
      firsts[layout.offsetOf((Enum<?>) field)] = cursor;
      cursor += ((CategoricalField) field).width();
    }
    for (int token = 0; token < count; token++) {
      System.arraycopy(firsts, 0, out, start + token * firsts.length, firsts.length);
    }
    return cursor;
  }

  /**
   * フィンガープリントのもとになる記述。アーキテクチャ、フィールドの並びと幅・尺度、テンソル、トークンの数、語彙、行動の番号の
   * ブロック、追加の入力のフィールド、区分、牌のコード、役のビットのまとめ方、待ちの数、公開の履歴の書き方を含む。
   */
  SchemaDigest describe() {
    SchemaDigest digest = SchemaDigest.named("epsilon-classic-input").with("architecture", architecture);
    for (FieldLayout<?> layout : layouts.values()) {
      digest.with(SchemaDigest.name(layout.type()), layout.describe());
    }
    StringJoiner tensors = new StringJoiner(", ", "[", "]");
    for (Tensor tensor : Tensor.values()) {
      if (tensor.in(architecture)) {
        tensors.add(tensor + "/" + tensor.slab() + "/" + tensor.axis() + ":" + width(tensor));
      }
    }
    digest.with("tensors", tensors);
    StringJoiner tokens = new StringJoiner(", ", "[", "]");
    for (StateToken token : StateToken.values()) {
      tokens.add(token + ":" + token.count);
    }
    digest.with("tokens", tokens);
    for (Class<?> vocabulary : SchemaDigest.sorted(ClassicVocabulary.class.getDeclaredClasses())) {
      if (vocabulary.isEnum()) {
        digest.with(SchemaDigest.name(vocabulary), Arrays.toString(vocabulary.getEnumConstants()));
      }
    }
    StringJoiner blocks = new StringJoiner(", ", "[", "]");
    for (ActionBlock block : ActionBlock.values()) {
      blocks.add(block + ":" + block.width());
    }
    return digest
        .with("actionBlocks", blocks)
        .values(Vocabulary.of(ClassicExtras.Field.class, Variant.TENHOU_4P))
        .with("buckets", Buckets.describe())
        .values(Vocabulary.of(TileFlag.class, Variant.TENHOU_4P))
        .with("tileCodes", TileFlag.CODES)
        .with("yakuChunks", YAKU_CHUNK_BITS + "x" + YAKU_CHUNKS)
        .with("waits", MAX_WAITS)
        .with("history", architecture.publicHistory() ? PublicHistory.describe() : "none");
  }

  private static Map<Architecture, ClassicSchema> schemas() {
    Map<Architecture, ClassicSchema> schemas = new EnumMap<>(Architecture.class);
    for (Architecture architecture : Architecture.values()) {
      schemas.put(architecture, new ClassicSchema(architecture));
    }
    return schemas;
  }

  /**
   * 1 つのフィールドの enum の要素を、どの位置に置くか。アーキテクチャが持つ定数だけを宣言の順に、それぞれの要素数ずつ並べる。
   * 不変なので、スレッド間で共有できる。
   */
  public static final class FieldLayout<E extends Enum<E> & InputField> {

    private final Class<E> type;
    private final List<E> fields;

    /** 定数の ordinal ごとの、先頭の要素の位置。アーキテクチャに無い定数は −1。 */
    private final int[] offsets;

    private final int stride;

    FieldLayout(Class<E> type, Architecture architecture) {
      this.type = type;
      E[] constants = type.getEnumConstants();
      offsets = new int[constants.length];
      List<E> present = new ArrayList<>();
      int cursor = 0;
      for (E field : constants) {
        if (field.in(architecture)) {
          present.add(field);
          offsets[field.ordinal()] = cursor;
          cursor += field.count();
        } else {
          offsets[field.ordinal()] = -1;
        }
      }
      fields = List.copyOf(present);
      stride = cursor;
    }

    public Class<E> type() {
      return type;
    }

    /** アーキテクチャが持つフィールド（宣言の順）。 */
    public List<E> fields() {
      return fields;
    }

    /** フィールドの先頭の要素の位置。アーキテクチャに無いフィールドは −1。 */
    public int offset(E field) {
      return offsets[field.ordinal()];
    }

    /** 1 トークンの要素数。 */
    public int stride() {
      return stride;
    }

    int offsetOf(Enum<?> field) {
      return offsets[field.ordinal()];
    }

    /** 各フィールドの名前、幅（整数のフィールド）か尺度（実数のフィールド）、要素数を並べた文字列。 */
    String describe() {
      StringJoiner text = new StringJoiner(", ", "[", "]");
      for (E field : fields) {
        String value =
            field instanceof CategoricalField categorical
                ? Integer.toString(categorical.width())
                : "/" + ((NumericField) field).scale();
        text.add(field + ":" + value + (field.count() == 1 ? "" : "x" + field.count()));
      }
      return text.toString();
    }
  }
}
