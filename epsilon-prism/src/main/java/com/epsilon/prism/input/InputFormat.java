package com.epsilon.prism.input;

import static com.epsilon.prism.input.InputSchema.UTILITY;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.codec.SchemaDigest;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.VariantScoped;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.FieldSpec.BoundedField;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Composition;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.CategoryField;
import com.epsilon.prism.input.InputSchema.DiscardExact;
import com.epsilon.prism.input.InputSchema.EventField;
import com.epsilon.prism.input.InputSchema.EventScalar;
import com.epsilon.prism.input.InputSchema.ExactField;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.RelationAttribute;
import com.epsilon.prism.input.InputSchema.RiichiExact;
import com.epsilon.prism.input.InputSchema.RoundField;
import com.epsilon.prism.input.InputSchema.RoundScalar;
import com.epsilon.prism.input.InputSchema.ScalarField;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.SeatScalar;
import com.epsilon.prism.input.InputSchema.TileField;
import com.epsilon.prism.input.InputSchema.TileKeys;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.InputSchema.TransitionScalar;
import com.epsilon.prism.input.InputSchema.WaitClass;
import com.epsilon.prism.input.InputSchema.WinExact;
import com.epsilon.prism.input.SuitCanonicalizer.SuitKey;
import com.epsilon.prism.input.TileSafety.Safety;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/** 対局ルールと入力機能に応じた列、語彙、符号と行ライターを提供する。 */
public final class InputFormat {
  private record Selection(Variant variant, Set<InputFeature> features) {}

  private static final Map<Selection, InputFormat> FORMATS = new ConcurrentHashMap<>();

  /** ルールの基本入力形式を返す。 */
  public static InputFormat of(Variant variant) {
    return InputSchema.of(variant);
  }

  /** 指定した入力機能を持つ、ルールごとの形式を返す。 */
  public static InputFormat of(Variant variant, Set<InputFeature> features) {
    if (features.isEmpty()) return of(variant);
    var selection = new Selection(variant, Set.copyOf(features));
    return FORMATS.computeIfAbsent(
        selection, key -> new InputFormat(key.variant(), key.features()));
  }

  public boolean has(InputFeature feature) {
    return features.contains(feature);
  }

  public CandidateCodes candidateCodes() {
    return candidateCodes;
  }

  public RowWriter writer() {
    return new Tokenizer(this);
  }

  /**
   * イベントのトークンの全ての列が公開の時点で確定するか（公開順の形式）。確定するなら、同じ経路・観測席の後の判断でも、公開済みの
   * イベントのバイトは変わらない（{@link HostBatch#eventPrefix}）。基本の形式は、判断ごとの色の正準化と、その席の今の位置を全ての
   * イベントに書く関係の列があるので確定しない。
   */
  public boolean eventsSettleOnPublication() {
    return has(InputFeature.PUBLIC_ORDER_EVENTS);
  }

  /**
   * 指紋を古い書き方で作るルール。配布済みの四人打ちの指紋（v714・v760）は、牌の並び・動かせる色・効用の値を指紋に足す前に作り、
   * {@link #MOVED} の型は {@code InputSchema} の中にあった頃の名前で書いた。この指紋を変えないため、このルールだけ古い書き方を続ける。
   * ほかのルール（三人打ちと、これから足すルール）は全てを書く。
   */
  private static final Variant LEGACY_DIGEST = Variant.TENHOU_4P;

  /** {@link #LEGACY_DIGEST} の指紋で、{@code InputSchema$型名} の形で書く型。 */
  private static final Set<Class<?>> MOVED =
      Set.of(
          Tensor.class,
          RowColumn.class,
          ActionColumn.class,
          PublicEvent.class,
          Safety.class,
          SuitKey.class,
          PassGrade.class);

  private final Variant variant;
  private final InputRules rules;
  private final Vocabulary<ActionRole> roles;
  private final Vocabulary<WinExact> winFields;
  private final KeyRange[] blocks = new KeyRange[KeyBlock.values().length];
  private final Keys keys;
  private final TileKeyCode tileCode;
  private final String description;
  private final String fingerprint;
  private final TokenLayout<RoundField> round;
  private final TokenLayout<SeatField> seat;
  private final TokenLayout<TileField> tile;
  private final TokenLayout<EventField> event;
  private final TokenLayout<TransitionField> transition;
  private final TokenLayout<RelationAttribute> relation;

  private final Set<InputFeature> features;
  private final CandidateCodes candidateCodes;

  InputFormat(Variant variant, Set<InputFeature> features) {
    this.features = features;
    this.variant = variant;
    rules = InputRules.of(variant);
    roles = Vocabulary.of(ActionRole.class, variant);
    winFields = Vocabulary.of(WinExact.class, variant);
    round = TokenLayout.of(RoundField.class, variant);
    seat = TokenLayout.of(SeatField.class, variant);
    tile = TokenLayout.of(TileField.class, variant);
    event = TokenLayout.of(EventField.class, variant);
    transition = TokenLayout.of(TransitionField.class, variant);
    relation = TokenLayout.of(RelationAttribute.class, variant);
    int firstKey = 0;
    int firstCode = 0;
    for (KeyBlock block : KeyBlock.values()) {
      var range = new Range(firstKey, firstCode, block.keys(variant), block.codes(variant));
      blocks[block.ordinal()] = range;
      firstKey += range.keys();
      firstCode += range.codes();
    }
    if (firstCode > FieldSpec.MAX_VOCABULARY)
      throw new IllegalStateException("Key codes exceed u8: " + firstCode);
    keys =
        new Keys(
            block(KeyBlock.ROUND).keys(),
            block(KeyBlock.SEAT).keys(),
            block(KeyBlock.TILE).keys(),
            firstCode);
    tileCode = new TileKeyCode(block(KeyBlock.TILE).firstCode(), TileKeys.SPACE);
    candidateCodes = new CandidateCodes(this);
    var digest = digest();
    for (InputFeature feature : InputFeature.values()) {
      if (has(feature)) feature.describe(digest, this);
    }
    description = digest.toString();
    fingerprint = digest.hex();
  }

  public String description() {
    return description;
  }

  public Vocabulary<WinExact> winFields() {
    return winFields;
  }

  public int roleIndex(ActionRole role) {
    return roles.index(role);
  }

  private record Range(int firstKey, int firstCode, int keys, int codes) implements KeyRange {}

  public KeyRange block(KeyBlock block) {
    return blocks[block.ordinal()];
  }

  public int seatCode(int seat, boolean involved) {
    return block(KeyBlock.SEAT).firstCode()
        + Objects.checkIndex(seat, variant.seats()) * 2
        + (involved ? 1 : 0);
  }

  public int tileCode(int count, boolean ukeire, WaitClass wait, boolean discarded) {
    return tileCode.encode(count, ukeire, wait == null ? 0 : wait.ordinal() + 1, discarded);
  }

  public Variant variant() {
    return variant;
  }

  public RankUtility utility() {
    return UTILITY;
  }

  public String fingerprint() {
    return fingerprint;
  }

  public TokenLayout<RoundField> round() {
    return round;
  }

  public TokenLayout<SeatField> seat() {
    return seat;
  }

  public TokenLayout<TileField> tile() {
    return tile;
  }

  public TokenLayout<EventField> event() {
    return event;
  }

  public TokenLayout<TransitionField> transition() {
    return transition;
  }

  public TokenLayout<RelationAttribute> relation() {
    return relation;
  }

  public int width(Tensor tensor) {
    return switch (tensor) {
      case ROUND_CATEGORY -> round.width();
      case ROUND_SCALAR -> RoundScalar.values().length;
      case SEAT_CATEGORY -> seat.width();
      case SEAT_SCALAR -> SeatScalar.values().length;
      case TILE_CATEGORY -> tile.width();
      case EVENT_CATEGORY -> event.width();
      case EVENT_SCALAR -> EventScalar.values().length;
      case RELATION -> relation.width();
      case TRANSITION_CATEGORY -> transition.width();
      case TRANSITION_SCALAR -> TransitionScalar.values().length;
      case TRANSITION_CODE -> candidateCodes.codeWidth();
      case DISCARD_EXACT -> DiscardExact.values().length;
      case ACTION_COLUMN -> ActionColumn.values().length;
      case ACTION_EXACT -> Math.max(winFields.size(), RiichiExact.values().length);
      case ROW_COLUMN -> RowColumn.values().length;
    };
  }

  public int eventCapacity() {
    return rules.eventCapacity;
  }

  public TileKeyCode tileKeyCode() {
    return tileCode;
  }

  public Keys keys() {
    return keys;
  }

  public List<ActionRole> roles() {
    return roles.constants();
  }

  public ActionRole role(Action.Type type) {
    if (!type.in(variant))
      throw new IllegalArgumentException("No role for " + type + " in " + variant);
    return ActionRole.of(type);
  }

  public ActionRole discard() {
    return ActionRole.DISCARD;
  }

  public ActionRole pass() {
    return ActionRole.PASS;
  }

  private SchemaDigest digest() {
    SchemaDigest digest = SchemaDigest.named(rules.schemaName);
    Map<Class<?>, Vocabulary<?>> valueTypes = new LinkedHashMap<>();
    for (Class<?> type : SchemaDigest.sorted(CategoryField.class.getPermittedSubclasses())) {
      appendBounded(digest, type, valueTypes);
    }
    appendBounded(digest, RelationAttribute.class, valueTypes);
    valueTypes.values().forEach(vocabulary -> values(digest, vocabulary));
    for (Class<?> type : SchemaDigest.sorted(ScalarField.class.getPermittedSubclasses())) {
      StringBuilder scalars = new StringBuilder();
      for (Object constant : type.getEnumConstants()) {
        if (constant instanceof VariantScoped scoped && !scoped.in(variant)) continue;
        scalars
            .append(constant)
            .append('/')
            .append(((ScalarField) constant).scale(variant))
            .append(',');
      }
      digest.with(SchemaDigest.name(type), scalars);
    }
    for (Class<?> type : SchemaDigest.sorted(ExactField.class.getPermittedSubclasses())) {
      StringBuilder exact = new StringBuilder();
      for (Object constant : type.getEnumConstants()) {
        if (constant instanceof VariantScoped scoped && !scoped.in(variant)) continue;
        exact.append(constant).append('/').append(((ExactField) constant).scale()).append(',');
      }
      digest.with(SchemaDigest.name(type), exact);
    }
    digest
        .values(Vocabulary.of(WaitClass.class, variant))
        .values(Vocabulary.of(ActionRole.class, variant));
    values(digest, Vocabulary.of(ActionColumn.class, variant));
    values(digest, Vocabulary.of(RowColumn.class, variant));
    values(digest, Vocabulary.of(SuitKey.class, variant));
    StringBuilder roles = new StringBuilder();
    for (Action.Type type : Vocabulary.of(Action.Type.class, variant).constants()) {
      roles.append(type).append('>').append(ActionRole.of(type)).append(',');
    }
    StringBuilder gates = new StringBuilder();
    for (ActionRole role : roles()) {
      gates.append(role).append('=').append(role.gate()).append(',');
    }
    StringBuilder tensors = new StringBuilder();
    for (Tensor tensor : Tensor.values()) {
      tensors.append(tensor).append(':').append(tensor.axis()).append(':');
      tensors.append(tensor.element()).append(',');
    }
    StringBuilder grades = new StringBuilder();
    for (PassGrade grade : PassGrade.values()) {
      grades.append(grade).append('=');
      if (grade.scored()) {
        grades.append(grade.basePoints());
      } else {
        grades.append('-');
      }
      grades.append(',');
    }
    StringBuilder blocks = new StringBuilder();
    for (KeyBlock block : KeyBlock.values()) {
      blocks.append(block).append('=').append(block(block).keys()).append('x');
      blocks.append(block(block).codes()).append(',');
    }
    digest
        .with("roles", roles)
        .with("gates", gates)
        .with(name(Tensor.class), tensors)
        .with(name(PassGrade.class), grades)
        .with(SchemaDigest.name(KeyBlock.class), blocks);
    StringBuilder utility = new StringBuilder(UTILITY.name());
    if (variant != LEGACY_DIGEST) {
      StringBuilder tiles = new StringBuilder();
      TileTokens tokens = tileTokens();
      for (int t = 0; t < tokens.count(); t++) tiles.append(tokens.kind(t)).append(',');
      digest.with("tiles", tiles).with("movableSuits", tokens.movableSuits());
      for (int rank = 0; rank < variant.seats(); rank++) {
        utility.append(rank == 0 ? '=' : ',').append(UTILITY.value(variant, rank));
      }
    }
    return digest
        .with("events", eventCapacity())
        .with("actions", Limits.MAX_LEGAL_ACTIONS)
        .with("transitions", Limits.MAX_TRANSITIONS)
        .with("alignment", BatchPlan.TENSOR_ALIGNMENT)
        .with("utility", utility);
  }

  /** 配布形式の記述で使う型名を返す。 */
  private String name(Class<?> type) {
    return variant == LEGACY_DIGEST && MOVED.contains(type)
        ? "InputSchema$" + type.getSimpleName()
        : SchemaDigest.name(type);
  }

  /** 語彙の型の名前と値の並びを書く（{@link SchemaDigest#values} と同じ形）。 */
  private void values(SchemaDigest digest, Vocabulary<?> vocabulary) {
    digest.with(name(vocabulary.type()), vocabulary);
  }

  /** フィールドごとに名前・書き方・語彙・繰り返しを書き、値の enum（このクラスのものも外のものも）を集める。 */
  private void appendBounded(
      SchemaDigest digest, Class<?> type, Map<Class<?>, Vocabulary<?>> valueTypes) {
    StringBuilder fields = new StringBuilder();
    for (Object constant : type.getEnumConstants()) {
      if (constant instanceof VariantScoped scoped && !scoped.in(variant)) continue;
      FieldSpec spec = ((BoundedField) constant).spec(variant);
      fields.append(constant).append(':').append(spec.coding()).append(':');
      fields.append(spec.vocabulary()).append('x').append(spec.repeats()).append(',');
      if (spec.values() != null) {
        valueTypes.putIfAbsent(spec.values().type(), spec.values());
      }
    }
    digest.with(SchemaDigest.name(type), fields);
  }

  /** 行ごとの席のトークンの数（席の数）。 */
  public int seats() {
    return keys().seat();
  }

  /** 行ごとの牌のトークンの数。 */
  public int tiles() {
    return keys().tile();
  }

  /** 行ごとに必ずあるトークン（局・席・牌）の数。 */
  public int fixedTokens() {
    return keys().count();
  }

  /** 行ごとのイベントのトークンの上限（公開イベントの上限 ＋ 槓ドラの表示牌の数）。 */
  public int maxEvents() {
    return variant().maxPublicEvents() + Limits.MAX_INDICATORS - 1;
  }

  /** 牌のトークンの並び。 */
  public TileTokens tileTokens() {
    return TileTokens.of(variant());
  }

  /**
   * 遷移ごとのキーの範囲。キーは局・席・牌の順に並ぶ。コードの番号は、局・席・牌の範囲をまたいで続けて振る。
   *
   * @param round 局のキーの数
   * @param seat 席のキーの数
   * @param tile 牌のキーの数
   * @param codes コードの総数（Ek・Ev の行の数）
   */
  public record Keys(int round, int seat, int tile, int codes) {

    /** キーの総数（遷移のコードの領域の列の数）。 */
    public int count() {
      return round + seat + tile;
    }
  }

  /** 局・席・牌のどれか 1 つのキーの範囲。 */
  public interface KeyRange {

    /** 範囲のキーの数。 */
    int keys();

    /** 範囲のコードの数。 */
    int codes();

    /** 範囲の最初のキーの列。 */
    int firstKey();

    /** 範囲の最初のコード。 */
    int firstCode();
  }

  /** BF16 で送る数値のフィールド。値は {@link #scale} で割ってから丸める。列は領域の中の {@link #ordinal} 番目。 */
  public interface Scalar {

    /** フィールドの領域（{@link Tensor#ROUND_SCALAR}・{@link Tensor#SEAT_SCALAR}・{@link Tensor#EVENT_SCALAR}・{@link Tensor#TRANSITION_SCALAR}）。 */
    Tensor tensor();

    /** 領域の中の列。 */
    int ordinal();

    /** 値を割る数。 */
    float scale(Variant variant);
  }

  /** FP32 のままヘッドへ渡す厳密なフィールド。値は {@link #scale} で割ってから送る。列は領域の中の {@link #ordinal} 番目。 */
  public interface Exact {

    /** フィールドの領域（{@link Tensor#DISCARD_EXACT} か {@link Tensor#ACTION_EXACT}）。 */
    Tensor tensor();

    /** 領域の中の列。 */
    int ordinal();

    /** 値を割る数。 */
    float scale();
  }

  /** 価値グラフでの手の扱い。番号は {@link InputFormat#roles()} の表の順。 */
  public interface Role {

    /** 表の中の番号。 */
    int ordinal();

    /** 扱いの名前。 */
    String name();

    /** ゲートのヘッドが値を出す手なら true。 */
    boolean gate();

    /** s の作り方。 */
    Composition composition();

    /** 局の終わり方が手で決まる（和了・九種九牌）なら true。 */
    boolean endsHand();
  }

  /** 1 行分の入力を書くライター。スレッドセーフではない。 */
  public interface RowWriter {

    /**
     * 1 行分の入力を、バッチの {@code row} 番目の行に書く。バッチの計画には、この行を {@code row} 番目に足してあること。
     *
     * @param o 行の判断の記録
     * @param batch 書き先のバッチ
     * @param row バッチの中の行の位置
     */
    void write(Observation o, HostInputs batch, int row);

    /**
     * {@link #write(Observation, HostInputs, int)} と同じ行を、同じ経路・観測席の前の判断の履歴の続きから書く。履歴の確定したイベントの
     * 行を写し、新しいイベントだけを符号化して、履歴を書き足す。履歴が O と合わなければ、全てを書いて履歴を作り直す。出力のバイトは
     * どちらでも同じ。
     *
     * @param history 履歴（null なら履歴を使わない）
     */
    default void write(Observation o, EventHistory history, HostInputs batch, int row) {
      write(o, batch, row);
    }

    /**
     * 物理の牌の種類を、色を正規化した順の牌のトークンに直す。色の置き換えは、直前に書いた行のものを使う。
     *
     * @param physicalTile 牌の種類
     * @return 色を正規化した牌のトークンの番号（{@link TileTokens}）
     */
    int canonicalTile(int physicalTile);
  }
}
