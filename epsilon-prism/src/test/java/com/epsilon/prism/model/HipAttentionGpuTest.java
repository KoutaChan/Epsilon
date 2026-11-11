package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.training.GradientCollector;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.RelationAttention;
import com.epsilon.prism.hip.RelationCodes;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.Random;
import java.util.zip.CRC32;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;
import org.testng.asserts.SoftAssert;

/**
 * HIP で自前に書いた Attention の演算（{@link HipAttention}）をテストする（rocm のグループ、GPU 1 枚）。{@link Case} の形ごとに、
 * 乱数で作った q・kv・出力の勾配（BF16）・マスク・関係のコード・ρ を入れて、次を確かめる。
 *
 * <ul>
 *   <li>djl-rocm の packedRelationScaledDotProductAttention と自動微分との比較: 出力は BF16 の一致の割合と差の最大、勾配（q・kv は
 *       BF16、ρ は行で足した FP32）はテンソルごとの相対 L2 の差を見る。
 *   <li>FP64 の参照（BF16 の入力をそのまま広げ、数式どおりの二重ループで計算する）との比較: BF16 の出力と勾配は、FP64 を BF16 に丸めた
 *       値とビットが同じ割合を見る。要素ごとには、出力が FP32 の出力を 1 回丸めた値で、FP32 の出力が桁落ちの幅 c·2⁻²⁴·Σ p|v| に収まる
 *       ことを見る（{@link #assertOutputElements}）。勾配は、1 ulp と rms·slack の幅か、桁落ちの幅に収まることを見る
 *       （{@link #assertGradientElements}）。どちらも尺度で割った比の分位を出す。FP32 の出力と ρ の勾配（行 × ヘッドの部分和）は相対の差、
 *       lse は絶対の差を見る。
 *   <li>1 つの検査が外れても、残りの検査と形は続ける。全ての形のログを出してから、外れた検査を形ごとにまとめて失敗にする。
 *   <li>同じ入力で順伝播・逆伝播を 2 回行い、全ての出力がビットまで同じこと。出力ごとの CRC32 もログに出し、別のプロセスで実行したログと
 *       比べられるようにする。
 * </ul>
 *
 * マスクには先頭からの長さのほかに、キーのタイルの中の穴（行 {@value #HOLE_ROW}）、途中のキーのタイルが丸ごと無い行（行
 * {@value #GAP_ROW}）、有効なキーが 1 つも無い行（行 {@value #EMPTY_ROW}。出力・lse・勾配が 0）を含む。
 * {@link #rowsAreIndependentOfTheBatch} は、一部の行だけのバッチや、パディングで S を広げたバッチでも、各行の結果が元のバッチとビットまで
 * 同じことを見る。{@link #deadQueriesContributeNothing} は、出力の勾配が 0 のクエリが q に依らず何にも寄与せず、クエリの範囲の外に
 * 出したときと同じ結果になることを見る。
 */
public class HipAttentionGpuTest {
  private static final Logger log = LoggerFactory.getLogger(HipAttentionGpuTest.class);

  private static final int ROWS = 48;
  private static final int HEADS = 8;
  private static final int FEATURES = RelationAttention.FEATURES;
  private static final int WIDTH = HEADS * FEATURES;
  private static final int WORDS = RelationCodes.WORDS;
  private static final float SCALE = 0.25f;
  private static final int LAST_QUERIES = 39;
  private static final int HOLE_ROW = 0;
  private static final int GAP_ROW = 1;
  private static final int EMPTY_ROW = 2;
  private static final int SUBSET_ROWS = 13;
  private static final int[] PADDED_TOKENS = {70, 80, 121, 160};
  /**
   * {@link #deadQueriesContributeNothing} で使う。途中に置く死んだクエリ（出力の勾配が 0 のクエリ）の範囲 [DEAD_FIRST, DEAD_END) と、
   * 先頭から数えた生きたクエリの数（これより後ろのクエリも死んだクエリにする）。
   */
  private static final int DEAD_FIRST = 36;
  private static final int DEAD_END = 44;
  private static final int LIVE_QUERIES = 60;
  private static final double FLOAT_UNIT = 0x1p-24;

  /**
   * FP32 の出力に許す誤差の幅の係数 c（{@link #assertOutputElements}）。和の側: 項 p_j v_j が t1 の和で通る丸めは多くて 4B + 6 回（B ≤ 10 個の
   * キーのタイルごとに WMMA 3 回と α の掛け算 1 回、WMMA の中の 5 段、最後の 1/Σ）。1 回の丸めは切り捨てでも相対 2⁻²³ なので、線形の
   * 上限は 92·2⁻²⁴·A。ℓ の側: ℓ₂ の丸め δ_j は p_j を相対で ln2·δ_j 動かし、o を Σ p_j (v_j − o)·ln2·δ_j だけ動かす
   * （|·| ≤ 2A·ln2·max δ）。このテストの大きさ（|ℓ₂| ≲ 10、Σ|τ| ≈ 4）では max δ ≈ 48·2⁻²⁴ で、線形の上限は約 66·2⁻²⁴·A。誤差は
   * 独立なので、実際はずっと小さい。c = 128 は 2 つの線形の上限の和（約 158）より小さく、t1 の順で足した FP32 の和の誤差より十分に大きい。
   * キーが 1 つ抜ける誤り（p_j|v_j| > 128·2⁻²⁴·A ≈ 7.6e-6·A）や P の誤りは幅を外れる。|o| ≈ A の要素では、幅は BF16 の ulp の 1/500
   * ほど。c は上限のしきい値で、分位（p99・p99.99）もログに出す。勾配も同じ c を勾配の尺度（{@link Reference}）に使う（dS の 1 段と、
   * キー・クエリのタイルの和で丸めは 7 回ほど増え、線形の上限は約 172）。
   */
  private static final int CANCELLATION = 128;

  /** 比べる形（クエリの数・キーの数・セグメントの数・セグメントあたりの値の数）。バックボーンの層は全てのトークンが、最後の層は先頭の 39 個だけがクエリになる。 */
  private enum Case {
    /** バックボーンの層、S = 70（キーのタイル 5 つ）。 */
    LAYER0_S70(70, 70, 7, 16),
    /** 最後の層、S = 70。 */
    LAYER1_S70(LAST_QUERIES, 70, 7, 16),
    /** キーのタイル 1 つ（逆伝播のワークグループは、キーのタイルの数より多い 5 ウェーブになる。t1 の段 B に 5 ウェーブ要るため）。 */
    LAYER0_S16(16, 16, 7, 16),
    /** キーのタイル 6 つ（キーのタイルの上限が 8 のカーネルを使い、LDS もその大きさになる）。 */
    LAYER0_S81(81, 81, 7, 16),
    /** キーのタイル 8 つ（上限が 8 のカーネルでいちばん大きい形）。 */
    LAYER0_S120(120, 120, 7, 16),
    /** 最後の層、S = 120。 */
    LAYER1_S120(LAST_QUERIES, 120, 7, 16),
    /** 行の長さの上限（キーのタイルの上限が 10 のカーネルを使う）。 */
    LAYER0_S160(160, 160, 7, 16),
    /** 最後の層、行の長さの上限。 */
    LAYER1_S160(LAST_QUERIES, 160, 7, 16),
    /** 8 セグメント・値 12 個（範囲を検査するコードパスを、外れずに通る）。 */
    SEGMENTS8_ENTRIES12(50, 50, 8, 12);

    private final int queries;
    private final int keys;
    private final int segments;
    private final int entries;

    Case(int queries, int keys, int segments, int entries) {
      this.queries = queries;
      this.keys = keys;
      this.segments = segments;
      this.entries = entries;
    }

    RelationAttention.Dimensions dimensions(int rows) {
      return new RelationAttention.Dimensions(rows, queries, keys, keys, HEADS, WORDS, segments, entries, SCALE);
    }
  }

  /** パディングしたクエリに入れる q。出力の勾配が 0 なので、どれを選んでも実際の行の出力は変わらない。 */
  private enum Padding {
    /** 正規乱数（正負が混じる）。 */
    GAUSSIAN,
    /** 正規乱数の絶対値。 */
    POSITIVE,
    ZERO
  }

  /** 演算の出力（BF16 は FP32 に広げた値）。 */
  private enum Output {
    OUT,
    OUT_FLOAT,
    LOG_SUM_EXP,
    GRAD_QUERY,
    GRAD_KEY_VALUE,
    GRAD_TABLE
  }

  /**
   * 演算の入力。q・kv・出力の勾配は、BF16 の値を FP32 に広げて持つ。{@code codes} はクエリとキーの組ごとの INT16 のワード
   * （[rows, codeQueries, keys, words]）で、{@code table} は ρ（[segments·entries, heads]）。
   */
  private record Inputs(
      RelationAttention.Dimensions d,
      float[] query,
      float[] keyValue,
      float[] mask,
      int[] codes,
      float[] table,
      float[] gradOut) {}

  /**
   * FP64 の参照の出力と、桁落ちを見るための要素ごとの尺度（FP32 の和の誤差は、項の絶対値の和に比例する）。OUT の尺度は A = Σ_j p_j |v_j|。
   * 勾配の尺度は |dS|ₐ = p_j·(Σ_d |do_d||v_jd| + Σ_d |do_d|·A_d)（dP − δ の両方の項の絶対値。δ の中の o の誤差も A に比例する）から求め、
   * dq は scale·Σ_j |dS|ₐ |k_j|、dk は scale·Σ_i |dS|ₐ |q_i|、dv は Σ_i p_ij |do_i|。
   */
  private record Reference(Map<Output, double[]> values, Map<Output, double[]> magnitudes) {}

  /** 1 回の実行の出力と、比べる行（入力の何行目か）。 */
  private record Placed(Map<Output, float[]> outputs, RelationAttention.Dimensions d, int[] rows) {}

  @FunctionalInterface
  private interface AttentionBody {
    void accept(NDManager manager, HipAttention attention);
  }

  @Test(groups = "rocm")
  public void kernelsMatchTheDjlOperatorAndTheReference() {
    StringBuilder failures = new StringBuilder();
    withAttention(
        (manager, attention) -> {
          for (Case shape : Case.values()) {
            Inputs inputs = generate(shape);
            Map<Output, float[]> first = run(manager, attention, inputs);
            Map<Output, float[]> second = run(manager, attention, inputs);
            printChecksums(shape, first);
            SoftAssert checks = new SoftAssert();
            for (Output output : Output.values()) {
              checks.assertTrue(
                  bitsEqual(first.get(output), second.get(output)), shape + " " + output + " differs between runs");
            }
            Reference fp64 = reference(inputs);
            compareWithDjl(checks, shape, inputs, first, djl(manager, inputs), fp64);
            compareWithReference(checks, shape, inputs, first, fp64);
            try {
              checks.assertAll();
            } catch (AssertionError e) {
              String message = e.getMessage().replace('\n', ' ');
              log.info("PRISM-HIPATTN-CASE-FAIL case={} {}", shape, message);
              failures.append('\n').append(message);
            }
          }
        });
    Assert.assertTrue(failures.isEmpty(), "cases outside the bounds:" + failures);
  }

  /**
   * 行の結果はバッチの組み方に依らない。S = 70 のバッチから 13 行だけを取ったバッチと、同じ行をパディングで S = 80・121・160 に広げた
   * バッチを作る。パディングしたキーは無効にし、パディングしたクエリは出力の勾配を 0、q を {@link Padding} のそれぞれにする。どのバッチも、
   * 実際の位置の全ての出力（ρ の勾配は行 × ヘッドの部分和）が元のバッチとビットまで同じであること。S = 80 ではパディングしたクエリが
   * タイル 4 の中だけにあり、121・160 では後ろにパディングだけのタイルが続く。
   */
  @Test(groups = "rocm")
  public void rowsAreIndependentOfTheBatch() {
    StringBuilder failures = new StringBuilder();
    withAttention(
        (manager, attention) -> {
          for (Case shape : new Case[] {Case.LAYER0_S70, Case.LAYER1_S70}) {
            Inputs whole = generate(shape);
            Map<Output, float[]> expected = run(manager, attention, whole);
            int[] rows = new int[SUBSET_ROWS];
            Arrays.setAll(rows, n -> n < 3 ? n : 3 * n - 3);
            for (int tokens : PADDED_TOKENS) {
              boolean paddedQueries = shape.queries == shape.keys && tokens > shape.keys;
              for (Padding padding : paddedQueries ? Padding.values() : new Padding[] {Padding.GAUSSIAN}) {
                Inputs part = pad(whole, rows, tokens, padding);
                compareRealPositions(
                    "ROWS",
                    String.format("case=%s rows=%d S=%d padding=%s", shape, SUBSET_ROWS, tokens, padding),
                    failures,
                    new Placed(expected, whole.d(), rows),
                    new Placed(run(manager, attention, part), part.d(), identity(SUBSET_ROWS)),
                    shape.queries,
                    shape.keys,
                    Output.values());
              }
            }
          }
        });
    Assert.assertTrue(failures.isEmpty(), "rows depend on the batch:" + failures);
  }

  /**
   * 死んだクエリ（ヘッドの出力の勾配が全て 0）は何にも寄与しない。LAYER0_S70 の全ての行で、クエリ [{@value #DEAD_FIRST},
   * {@value #DEAD_END})（タイル 2 の中）と {@value #LIVE_QUERIES} 以降（タイル 3 の後ろとタイル 4）の出力の勾配を 0 にしたバッチが、
   * 次を満たす。
   *
   * <ul>
   *   <li>死んだクエリの q を 0 にしたバッチと、全ての勾配がビットまで同じ。
   *   <li>クエリを先頭の {@value #LIVE_QUERIES} 個にしたバッチ（後ろの死んだクエリを範囲の外に出す）と、実際の位置の全ての出力が
   *       ビットまで同じ。
   *   <li>死んだクエリの dq が +0（ビットまで 0）。
   * </ul>
   */
  @Test(groups = "rocm")
  public void deadQueriesContributeNothing() {
    StringBuilder failures = new StringBuilder();
    withAttention(
        (manager, attention) -> {
          Inputs base = generate(Case.LAYER0_S70);
          RelationAttention.Dimensions d = base.d();
          int queries = d.queries();
          boolean[] dead = new boolean[queries];
          Arrays.fill(dead, DEAD_FIRST, DEAD_END, true);
          Arrays.fill(dead, LIVE_QUERIES, queries, true);
          float[] gradOut = base.gradOut().clone();
          float[] zeroQuery = base.query().clone();
          float[] liveQuery = new float[ROWS * LIVE_QUERIES * WIDTH];
          float[] liveGradOut = new float[liveQuery.length];
          for (int r = 0; r < ROWS; r++) {
            for (int i = 0; i < queries; i++) {
              int at = (r * queries + i) * WIDTH;
              if (dead[i]) {
                Arrays.fill(gradOut, at, at + WIDTH, 0f);
                Arrays.fill(zeroQuery, at, at + WIDTH, 0f);
              }
            }
            int from = r * queries * WIDTH;
            int to = r * LIVE_QUERIES * WIDTH;
            System.arraycopy(base.query(), from, liveQuery, to, LIVE_QUERIES * WIDTH);
            System.arraycopy(gradOut, from, liveGradOut, to, LIVE_QUERIES * WIDTH);
          }
          RelationAttention.Dimensions live =
              new RelationAttention.Dimensions(
                  ROWS, LIVE_QUERIES, d.keys(), d.codeQueries(), HEADS, WORDS, d.segments(), d.entries(), SCALE);
          Inputs whole = new Inputs(d, base.query(), base.keyValue(), base.mask(), base.codes(), base.table(), gradOut);
          Map<Output, float[]> expected = run(manager, attention, whole);
          Map<Output, float[]> zeroed =
              run(
                  manager,
                  attention,
                  new Inputs(d, zeroQuery, base.keyValue(), base.mask(), base.codes(), base.table(), gradOut));
          Map<Output, float[]> truncated =
              run(
                  manager,
                  attention,
                  new Inputs(live, liveQuery, base.keyValue(), base.mask(), base.codes(), base.table(), liveGradOut));
          int[] rows = identity(ROWS);
          compareRealPositions(
              "DEAD",
              "case=LAYER0_S70 variant=zero-q",
              failures,
              new Placed(expected, d, rows),
              new Placed(zeroed, d, rows),
              queries,
              d.keys(),
              Output.GRAD_QUERY,
              Output.GRAD_KEY_VALUE,
              Output.GRAD_TABLE);
          compareRealPositions(
              "DEAD",
              "case=LAYER0_S70 variant=queries-" + LIVE_QUERIES,
              failures,
              new Placed(expected, d, rows),
              new Placed(truncated, live, rows),
              LIVE_QUERIES,
              d.keys(),
              Output.values());
          float[] gradQuery = expected.get(Output.GRAD_QUERY);
          int elements = 0;
          int nonzero = 0;
          for (int r = 0; r < ROWS; r++) {
            for (int i = 0; i < queries; i++) {
              for (int f = 0; dead[i] && f < WIDTH; f++) {
                elements++;
                nonzero += Float.floatToRawIntBits(gradQuery[(r * queries + i) * WIDTH + f]) == 0 ? 0 : 1;
              }
            }
          }
          log.info("PRISM-HIPATTN-DEAD case=LAYER0_S70 dq of dead queries +0: {} of {}", elements - nonzero, elements);
          if (nonzero > 0) {
            failures.append(String.format("%n%d dq elements of dead queries are not +0", nonzero));
          }
        });
    Assert.assertTrue(failures.isEmpty(), "dead queries contribute:" + failures);
  }

  /** GPU 0 のストリームの上で演算を作り、{@code body} に渡す。GPU が無ければテストを飛ばす。 */
  private static void withAttention(AttentionBody body) {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    Device gpu = Device.gpu(0);
    try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
        PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
        PtStreamScope ignored = stream.openScope()) {
      body.accept(manager, HipAttention.on(HipLibrary.load(), stream));
    }
  }

  /** 形 {@code shape} の入力を乱数で作る。乱数の種は形で決まる。 */
  private static Inputs generate(Case shape) {
    Random random = new Random(17 + shape.ordinal());
    RelationAttention.Dimensions d = shape.dimensions(ROWS);
    int keys = d.keys();
    float[] query = bfloat16Gaussian(random, ROWS * d.queries() * WIDTH, 1f);
    float[] keyValue = bfloat16Gaussian(random, ROWS * keys * 2 * WIDTH, 1f);
    float[] mask = new float[ROWS * keys];
    int shortest = Math.min(LAST_QUERIES, keys);
    for (int r = 0; r < ROWS; r++) {
      int tokens =
          switch (r) {
            case EMPTY_ROW -> 0;
            case HOLE_ROW, GAP_ROW -> keys;
            default -> shortest + random.nextInt(keys - shortest + 1);
          };
      Arrays.fill(mask, r * keys, r * keys + tokens, 1f);
    }
    // 穴の行はキーのタイル 0 の中に穴を空け、抜けたタイルの行は後ろのタイルを残したままタイル 1 を丸ごと無効にする
    Arrays.fill(mask, HOLE_ROW * keys + 5, HOLE_ROW * keys + 10, 0f);
    if (keys >= 48) {
      Arrays.fill(mask, GAP_ROW * keys + 16, GAP_ROW * keys + 32, 0f);
    }
    int[] codes = new int[ROWS * keys * keys * WORDS];
    for (int pair = 0; pair < ROWS * keys * keys; pair++) {
      int low = 0;
      int high = 0;
      for (int s = 0; s < d.segments(); s++) {
        int code = random.nextInt(d.entries());
        if (s < 4) {
          low |= code << (4 * s);
        } else {
          high |= code << (4 * (s - 4));
        }
      }
      codes[pair * WORDS] = (short) low;
      codes[pair * WORDS + 1] = (short) high;
    }
    float[] table = gaussian(random, d.tableRows() * HEADS, 0.5f);
    float[] gradOut = bfloat16Gaussian(random, ROWS * d.queries() * WIDTH, 1f);
    return new Inputs(d, query, keyValue, mask, codes, table, gradOut);
  }

  /**
   * {@code whole} から行 {@code rows} を取り出し、キー（バックボーンの層ならクエリも）を {@code tokens} 個までパディングして並べた入力を
   * 作る。パディングしたキーは無効で、値は乱数、コードは 0。パディングしたクエリは q が {@code padding} の値で、出力の勾配が 0。
   */
  private static Inputs pad(Inputs whole, int[] rows, int tokens, Padding padding) {
    RelationAttention.Dimensions w = whole.d();
    int queries = w.queries() == w.keys() ? tokens : w.queries();
    RelationAttention.Dimensions d =
        new RelationAttention.Dimensions(
            rows.length, queries, tokens, tokens, HEADS, WORDS, w.segments(), w.entries(), SCALE);
    Random random = new Random(tokens);
    float[] query = bfloat16Gaussian(random, rows.length * queries * WIDTH, 1f);
    for (int i = 0; i < query.length; i++) {
      query[i] =
          switch (padding) {
            case GAUSSIAN -> query[i];
            case POSITIVE -> Math.abs(query[i]);
            case ZERO -> 0f;
          };
    }
    float[] keyValue = bfloat16Gaussian(random, rows.length * tokens * 2 * WIDTH, 1f);
    float[] mask = new float[rows.length * tokens];
    int[] codes = new int[rows.length * tokens * tokens * WORDS];
    float[] gradOut = new float[rows.length * queries * WIDTH];
    for (int n = 0; n < rows.length; n++) {
      int r = rows[n];
      System.arraycopy(whole.query(), r * w.queries() * WIDTH, query, n * queries * WIDTH, w.queries() * WIDTH);
      System.arraycopy(whole.gradOut(), r * w.queries() * WIDTH, gradOut, n * queries * WIDTH, w.queries() * WIDTH);
      System.arraycopy(
          whole.keyValue(), r * w.keys() * 2 * WIDTH, keyValue, n * tokens * 2 * WIDTH, w.keys() * 2 * WIDTH);
      System.arraycopy(whole.mask(), r * w.keys(), mask, n * tokens, w.keys());
      for (int i = 0; i < w.codeQueries(); i++) {
        System.arraycopy(
            whole.codes(),
            (r * w.codeQueries() + i) * w.keys() * WORDS,
            codes,
            (n * tokens + i) * tokens * WORDS,
            w.keys() * WORDS);
      }
    }
    return new Inputs(d, query, keyValue, mask, codes, whole.table(), gradOut);
  }

  /** 自前の演算で順伝播と逆伝播を行い、全ての出力を返す。 */
  private static Map<Output, float[]> run(NDManager manager, HipAttention attention, Inputs in) {
    RelationAttention.Dimensions d = in.d();
    Shape queries = new Shape(d.rows(), d.queries(), WIDTH);
    Shape keys = new Shape(d.rows(), d.keys(), 2 * WIDTH);
    try (NDManager scope = manager.newSubManager()) {
      NDArray query = bfloat16(scope, in.query(), queries);
      NDArray keyValue = bfloat16(scope, in.keyValue(), keys);
      NDArray mask = scope.create(in.mask(), new Shape(d.rows(), d.keys()));
      NDArray codes = codes(scope, in);
      NDArray table = scope.create(in.table(), new Shape(d.tableRows(), HEADS));
      NDArray out = scope.create(queries, DataType.BFLOAT16);
      NDArray outFloat = scope.create(queries, DataType.FLOAT32);
      NDArray logSumExp = scope.create(new Shape(d.rows(), HEADS, d.queries()), DataType.FLOAT32);
      attention.forward(d, query, keyValue, mask, codes, table, out, outFloat, logSumExp);
      NDArray gradQuery = scope.create(queries, DataType.BFLOAT16);
      NDArray gradKeyValue = scope.create(keys, DataType.BFLOAT16);
      NDArray gradTable = scope.create(new Shape((long) d.rows() * HEADS, d.tableRows()), DataType.FLOAT32);
      attention.backward(
          d,
          query,
          keyValue,
          mask,
          codes,
          table,
          outFloat,
          logSumExp,
          bfloat16(scope, in.gradOut(), queries),
          gradQuery,
          gradKeyValue,
          gradTable);
      Map<Output, float[]> outputs = new EnumMap<>(Output.class);
      outputs.put(Output.OUT, widen(out));
      outputs.put(Output.OUT_FLOAT, outFloat.toFloatArray());
      outputs.put(Output.LOG_SUM_EXP, logSumExp.toFloatArray());
      outputs.put(Output.GRAD_QUERY, widen(gradQuery));
      outputs.put(Output.GRAD_KEY_VALUE, widen(gradKeyValue));
      outputs.put(Output.GRAD_TABLE, gradTable.toFloatArray());
      return outputs;
    }
  }

  /**
   * djl-rocm の演算を自動微分で動かす（関係の表のうち、値に足す側の列は 0 にする）。有効なキーが無い行は djl の演算が扱えないので、
   * キー 0 を有効にし、出力の勾配を 0 にする。その行の勾配の寄与は 0 になり、自前の演算の P = 0 と同じになる。その行の出力だけは比べない。
   * ρ の勾配は [segments·entries, heads]。
   */
  private static Map<Output, float[]> djl(NDManager manager, Inputs in) {
    RelationAttention.Dimensions d = in.d();
    Shape queries = new Shape(d.rows(), d.queries(), WIDTH);
    float[] mask = in.mask().clone();
    mask[EMPTY_ROW * d.keys()] = 1f;
    float[] gradOut = in.gradOut().clone();
    Arrays.fill(gradOut, EMPTY_ROW * d.queries() * WIDTH, (EMPTY_ROW + 1) * d.queries() * WIDTH, 0f);
    try (NDManager scope = manager.newSubManager()) {
      NDArray q = bfloat16(scope, in.query(), queries);
      NDArray kv = bfloat16(scope, in.keyValue(), new Shape(d.rows(), d.keys(), 2 * WIDTH));
      NDArray full =
          scope
              .create(in.table(), new Shape(d.tableRows(), HEADS))
              .concat(scope.zeros(new Shape(d.tableRows(), WIDTH)), 1);
      q.setRequiresGradient(true);
      kv.setRequiresGradient(true);
      full.setRequiresGradient(true);
      NDArray codes = codes(scope, in);
      NDArray pairs = d.queries() == d.codeQueries() ? codes : codes.get(new NDIndex(":, 0:{}", d.queries()));
      NDArray expected;
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        expected =
            NDArrays.packedRelationScaledDotProductAttention(
                q, kv, scope.create(mask, new Shape(d.rows(), d.keys())), pairs, full, HEADS, d.entries(), SCALE);
        collector.backward(expected.mul(bfloat16(scope, gradOut, queries)).sum());
      }
      Map<Output, float[]> outputs = new EnumMap<>(Output.class);
      outputs.put(Output.OUT, widen(expected));
      outputs.put(Output.GRAD_QUERY, widen(q.getGradient()));
      outputs.put(Output.GRAD_KEY_VALUE, widen(kv.getGradient()));
      outputs.put(Output.GRAD_TABLE, full.getGradient().get(new NDIndex(":, 0:{}", HEADS)).toFloatArray());
      return outputs;
    }
  }

  /**
   * FP64 の参照（数式どおり。BF16 の入力をそのまま広げる）: ℓ_ij = (q_i·k_j)·scale + Σ_s ρ[entries·s + c_s(i, j), h]（有効なキー
   * だけ）、p = softmax(ℓ)、o = Σ p v、lse = log Σ exp ℓ、δ_i = do_i·o_i、ds_ij = p_ij·(do_i·v_j − δ_i)、dq = scale·Σ ds k、
   * dk = scale·Σ ds q、dv = Σ p do、ρ の勾配は行 × ヘッドごとの Σ ds。有効なキーが無いクエリは o = 0、lse = 0、勾配 0。
   */
  private static Reference reference(Inputs in) {
    RelationAttention.Dimensions d = in.d();
    int queries = d.queries();
    int keys = d.keys();
    int tableRows = d.tableRows();
    double[] out = new double[ROWS * queries * WIDTH];
    double[] magnitude = new double[out.length];
    double[] logSumExp = new double[ROWS * HEADS * queries];
    double[] gradQuery = new double[out.length];
    double[] gradQueryMagnitude = new double[out.length];
    double[] gradKeyValue = new double[ROWS * keys * 2 * WIDTH];
    double[] gradKeyValueMagnitude = new double[gradKeyValue.length];
    double[] gradTable = new double[ROWS * HEADS * tableRows];
    double[] logit = new double[keys];
    double[] probability = new double[keys];
    int[][] entry = new int[keys][d.segments()];
    double[] o = new double[FEATURES];
    for (int r = 0; r < ROWS; r++) {
      for (int h = 0; h < HEADS; h++) {
        int column = h * FEATURES;
        int tableRow = (r * HEADS + h) * tableRows;
        for (int i = 0; i < queries; i++) {
          int queryAt = (r * queries + i) * WIDTH + column;
          double max = Double.NEGATIVE_INFINITY;
          for (int j = 0; j < keys; j++) {
            if (in.mask()[r * keys + j] == 0f) {
              continue;
            }
            int keyAt = (r * keys + j) * 2 * WIDTH + column;
            double dot = 0;
            for (int f = 0; f < FEATURES; f++) {
              dot += (double) in.query()[queryAt + f] * in.keyValue()[keyAt + f];
            }
            double bias = 0;
            for (int s = 0; s < d.segments(); s++) {
              entry[j][s] = s * d.entries() + code(in, r, i, j, s);
              bias += in.table()[entry[j][s] * HEADS + h];
            }
            logit[j] = dot * SCALE + bias;
            max = Math.max(max, logit[j]);
          }
          if (max == Double.NEGATIVE_INFINITY) {
            continue;
          }
          double sum = 0;
          for (int j = 0; j < keys; j++) {
            probability[j] = in.mask()[r * keys + j] == 0f ? 0 : Math.exp(logit[j] - max);
            sum += probability[j];
          }
          logSumExp[(r * HEADS + h) * queries + i] = max + Math.log(sum);
          Arrays.fill(o, 0);
          for (int j = 0; j < keys; j++) {
            probability[j] /= sum;
            int valueAt = (r * keys + j) * 2 * WIDTH + WIDTH + column;
            for (int f = 0; f < FEATURES; f++) {
              o[f] += probability[j] * in.keyValue()[valueAt + f];
              magnitude[queryAt + f] += probability[j] * Math.abs(in.keyValue()[valueAt + f]);
            }
          }
          double delta = 0;
          double deltaMagnitude = 0;
          for (int f = 0; f < FEATURES; f++) {
            out[queryAt + f] = o[f];
            delta += in.gradOut()[queryAt + f] * o[f];
            deltaMagnitude += Math.abs(in.gradOut()[queryAt + f]) * magnitude[queryAt + f];
          }
          for (int j = 0; j < keys; j++) {
            if (in.mask()[r * keys + j] == 0f) {
              continue;
            }
            int keyAt = (r * keys + j) * 2 * WIDTH + column;
            int valueAt = keyAt + WIDTH;
            double gradProbability = 0;
            double gradProbabilityMagnitude = 0;
            for (int f = 0; f < FEATURES; f++) {
              gradProbability += (double) in.gradOut()[queryAt + f] * in.keyValue()[valueAt + f];
              gradProbabilityMagnitude += Math.abs((double) in.gradOut()[queryAt + f] * in.keyValue()[valueAt + f]);
            }
            double gradScore = probability[j] * (gradProbability - delta);
            double gradScoreMagnitude = probability[j] * (gradProbabilityMagnitude + deltaMagnitude);
            for (int f = 0; f < FEATURES; f++) {
              gradQuery[queryAt + f] += SCALE * gradScore * in.keyValue()[keyAt + f];
              gradKeyValue[keyAt + f] += SCALE * gradScore * in.query()[queryAt + f];
              gradKeyValue[valueAt + f] += probability[j] * in.gradOut()[queryAt + f];
              gradQueryMagnitude[queryAt + f] += SCALE * gradScoreMagnitude * Math.abs(in.keyValue()[keyAt + f]);
              gradKeyValueMagnitude[keyAt + f] += SCALE * gradScoreMagnitude * Math.abs(in.query()[queryAt + f]);
              gradKeyValueMagnitude[valueAt + f] += probability[j] * Math.abs(in.gradOut()[queryAt + f]);
            }
            for (int s = 0; s < d.segments(); s++) {
              gradTable[tableRow + entry[j][s]] += gradScore;
            }
          }
        }
      }
    }
    Map<Output, double[]> outputs = new EnumMap<>(Output.class);
    outputs.put(Output.OUT, out);
    outputs.put(Output.OUT_FLOAT, out);
    outputs.put(Output.LOG_SUM_EXP, logSumExp);
    outputs.put(Output.GRAD_QUERY, gradQuery);
    outputs.put(Output.GRAD_KEY_VALUE, gradKeyValue);
    outputs.put(Output.GRAD_TABLE, gradTable);
    Map<Output, double[]> magnitudes = new EnumMap<>(Output.class);
    magnitudes.put(Output.OUT, magnitude);
    magnitudes.put(Output.GRAD_QUERY, gradQueryMagnitude);
    magnitudes.put(Output.GRAD_KEY_VALUE, gradKeyValueMagnitude);
    return new Reference(outputs, magnitudes);
  }

  /** 行 r でクエリ i とキー j を組にしたときの、セグメント s のコードを返す。 */
  private static int code(Inputs in, int r, int i, int j, int s) {
    RelationAttention.Dimensions d = in.d();
    int word = in.codes()[((r * d.codeQueries() + i) * d.keys() + j) * WORDS + s / 4] & 0xffff;
    return (word >> (4 * (s % 4))) & 15;
  }

  /**
   * djl-rocm の演算と比べる（出力は、有効なキーが無い行を除いて比べる）。式の取り違えを見つける粗い検査で、精度は FP64 の参照と要素ごとに見る
   * （{@link #compareWithReference}）。BF16 の一致のしきい値は 0.999（FP64 と比べるときと同じ）。精度が同じくらいの 2 つの FP32 の和
   * でも、足す順が違うと BF16 の丸めの境界に近い要素で食い違い、その割合はキーの数とともに増える。ログの行には、FP64 を BF16 に丸めた
   * 値との食い違いの内訳（hip だけ違う・djl だけ違う・両方違う）も出す。
   */
  private static void compareWithDjl(
      SoftAssert checks,
      Case shape,
      Inputs in,
      Map<Output, float[]> actual,
      Map<Output, float[]> expected,
      Reference fp64) {
    RelationAttention.Dimensions d = in.d();
    float[] expectedOut = expected.get(Output.OUT);
    float[] actualOut = actual.get(Output.OUT);
    double[] referenceOut = fp64.values().get(Output.OUT);
    int emptyFirst = EMPTY_ROW * d.queries() * WIDTH;
    int emptyEnd = emptyFirst + d.queries() * WIDTH;
    int same = 0;
    int hipOnly = 0;
    int djlOnly = 0;
    int both = 0;
    double maximum = 0;
    double largest = 0;
    for (int i = 0; i < expectedOut.length; i++) {
      if (i < emptyFirst || i >= emptyEnd) {
        int rounded = Float.floatToRawIntBits(bfloat16(referenceOut[i]));
        boolean hipRight = Float.floatToRawIntBits(actualOut[i]) == rounded;
        boolean djlRight = Float.floatToRawIntBits(expectedOut[i]) == rounded;
        same += Float.floatToRawIntBits(expectedOut[i]) == Float.floatToRawIntBits(actualOut[i]) ? 1 : 0;
        hipOnly += !hipRight && djlRight ? 1 : 0;
        djlOnly += hipRight && !djlRight ? 1 : 0;
        both += !hipRight && !djlRight ? 1 : 0;
        maximum = Math.max(maximum, Math.abs(expectedOut[i] - actualOut[i]));
        largest = Math.max(largest, Math.abs(expectedOut[i]));
      }
    }
    int compared = expectedOut.length - (emptyEnd - emptyFirst);
    double sameShare = same / (double) compared;
    double queryError = relative(expected.get(Output.GRAD_QUERY), actual.get(Output.GRAD_QUERY));
    double keyValueError = relative(expected.get(Output.GRAD_KEY_VALUE), actual.get(Output.GRAD_KEY_VALUE));
    double tableError = relative(expected.get(Output.GRAD_TABLE), rowSum(actual.get(Output.GRAD_TABLE), d));
    log.info(
        "PRISM-HIPATTN case={} queries={} keys={} out same={} max|d|={} gradients rel q={} kv={}"
            + " rho={} fp64 same hip={} djl={} wrong hip-only={} djl-only={} both={} of {}",
        shape,
        d.queries(),
        d.keys(),
        String.format("%.5f", sameShare),
        String.format("%.3g", maximum),
        String.format("%.3g", queryError),
        String.format("%.3g", keyValueError),
        String.format("%.3g", tableError),
        String.format("%.5f", 1 - (hipOnly + both) / (double) compared),
        String.format("%.5f", 1 - (djlOnly + both) / (double) compared),
        hipOnly,
        djlOnly,
        both,
        compared);
    checks.assertTrue(sameShare >= 0.999, shape + " output bits against djl " + sameShare);
    checks.assertTrue(maximum <= bfloat16Ulp(largest), shape + " output max|d| " + maximum);
    checks.assertTrue(queryError <= 1e-4, shape + " query gradient " + queryError);
    checks.assertTrue(keyValueError <= 1e-4, shape + " key/value gradient " + keyValueError);
    checks.assertTrue(tableError <= 5e-6, shape + " relation table gradient " + tableError);
  }

  /** FP64 の参照と比べる。有効なキーが無い行は出力・lse・勾配が 0 で、どの出力にも NaN・Inf が無いことも確かめる。 */
  private static void compareWithReference(
      SoftAssert checks, Case shape, Inputs in, Map<Output, float[]> actual, Reference fp64) {
    RelationAttention.Dimensions d = in.d();
    Map<Output, double[]> reference = fp64.values();
    for (Output output : Output.values()) {
      int infinite = 0;
      for (float value : actual.get(output)) {
        infinite += Float.isFinite(value) ? 0 : 1;
      }
      checks.assertEquals(infinite, 0, shape + " " + output + " elements not finite");
    }
    assertOutputElements(checks, shape, in, actual, reference.get(Output.OUT), fp64.magnitudes().get(Output.OUT));
    double outShare = bfloat16Share(actual.get(Output.OUT), reference.get(Output.OUT));
    double queryShare = assertGradientElements(checks, shape, Output.GRAD_QUERY, actual, fp64, 1e-5);
    double keyValueShare = assertGradientElements(checks, shape, Output.GRAD_KEY_VALUE, actual, fp64, 1e-5);
    double outFloatError = relative(reference.get(Output.OUT_FLOAT), actual.get(Output.OUT_FLOAT));
    double tableError = relative(reference.get(Output.GRAD_TABLE), actual.get(Output.GRAD_TABLE));
    double tableWorst = worst(reference.get(Output.GRAD_TABLE), actual.get(Output.GRAD_TABLE));
    double[] lse = reference.get(Output.LOG_SUM_EXP);
    float[] actualLse = actual.get(Output.LOG_SUM_EXP);
    double lseWorst = 0;
    for (int i = 0; i < lse.length; i++) {
      lseWorst = Math.max(lseWorst, Math.abs(actualLse[i] - lse[i]) / (1 + Math.abs(lse[i])));
    }
    log.info(
        "PRISM-HIPATTN-FP64 case={} out same={} outFloat rel={} lse={} q same={} kv same={} rho rel={} worst={}",
        shape,
        String.format("%.5f", outShare),
        String.format("%.3g", outFloatError),
        String.format("%.3g", lseWorst),
        String.format("%.5f", queryShare),
        String.format("%.5f", keyValueShare),
        String.format("%.3g", tableError),
        String.format("%.3g", tableWorst));
    checks.assertTrue(outShare >= 0.999, shape + " output bits against FP64 " + outShare);
    checks.assertTrue(queryShare >= 0.995, shape + " query gradient bits against FP64 " + queryShare);
    checks.assertTrue(keyValueShare >= 0.995, shape + " key/value gradient bits against FP64 " + keyValueShare);
    checks.assertTrue(outFloatError <= 1e-6, shape + " FP32 output against FP64 " + outFloatError);
    checks.assertTrue(lseWorst <= 2e-6, shape + " log-sum-exp against FP64 " + lseWorst);
    checks.assertTrue(tableError <= 1e-5, shape + " relation table gradient against FP64 " + tableError);
    checks.assertTrue(tableWorst <= 1e-5, shape + " relation table gradient element against FP64 " + tableWorst);

    int queryFirst = EMPTY_ROW * d.queries() * WIDTH;
    for (Output output : new Output[] {Output.OUT, Output.OUT_FLOAT, Output.GRAD_QUERY}) {
      assertZero(checks, shape, output, actual.get(output), queryFirst, d.queries() * WIDTH);
    }
    int lseFirst = EMPTY_ROW * HEADS * d.queries();
    assertZero(checks, shape, Output.LOG_SUM_EXP, actual.get(Output.LOG_SUM_EXP), lseFirst, HEADS * d.queries());
    int keyFirst = EMPTY_ROW * d.keys() * 2 * WIDTH;
    assertZero(
        checks, shape, Output.GRAD_KEY_VALUE, actual.get(Output.GRAD_KEY_VALUE), keyFirst, d.keys() * 2 * WIDTH);
  }

  /** 有効なキーが無い行で、出力 {@code output} の [first, first + length) が 0 であることを確かめる。 */
  private static void assertZero(
      SoftAssert checks, Case shape, Output output, float[] values, int first, int length) {
    int nonzero = 0;
    for (int i = first; i < first + length; i++) {
      nonzero += values[i] == 0f ? 0 : 1;
    }
    checks.assertEquals(nonzero, 0, shape + " " + output + " elements of the row without keys not 0");
  }

  /**
   * 出力を要素ごとに検査する。o は FP32 の和の結果 o₃₂（OUT_FLOAT）を BF16 に 1 回丸めた値なので、2 つに分けて見る。
   *
   * <ul>
   *   <li>1 回の丸め: OUT が bf16(OUT_FLOAT) とビットまで同じ。
   *   <li>FP32 の和の幅: |o₃₂ − o₆₄| ≤ c·2⁻²⁴·A、A = Σ_j p_j |v_j|（c = {@value #CANCELLATION}）。和の丸めの誤差は項の絶対値の和に
   *       比例するので、桁落ちで |o| ≪ A になる要素（例: |o| ≈ 2e-5、A ≈ 0.96）でも、FP32 と同じ精度かを見られる。
   * </ul>
   *
   * 2 つを合わせると |o − o₆₄| ≤ c·2⁻²⁴·A +（o₃₂ の BF16 の半 ulp）。比 |o₃₂ − o₆₄|/(2⁻²⁴·A) の分位（A > 0 の要素）を、全体と、行の
   * 種類（穴・抜けたタイル）ごと、有効なキーを含むタイルの数 B ごとに出す（c より下の小さな系統誤差、例えばタイルの数とともに増える
   * α の順の誤りを分位で見る）。外れた要素は検査に記録する前に全て数え、先頭の幾つかを A と一緒に出す。
   */
  private static void assertOutputElements(
      SoftAssert checks, Case shape, Inputs in, Map<Output, float[]> actual, double[] expected, double[] magnitude) {
    RelationAttention.Dimensions d = in.d();
    float[] out = actual.get(Output.OUT);
    float[] outFloat = actual.get(Output.OUT_FLOAT);
    int[] rowStratum = rowStrata(in);
    double[] ratios = new double[out.length];
    int[] strata = new int[out.length];
    int count = 0;
    int unrounded = 0;
    int outside = 0;
    double worst = 0;
    int worstAt = 0;
    StringBuilder first = new StringBuilder();
    for (int i = 0; i < out.length; i++) {
      if (Float.floatToRawIntBits(out[i]) != Float.floatToRawIntBits(bfloat16(outFloat[i]))) {
        unrounded++;
      }
      if (magnitude[i] == 0) {
        continue;
      }
      double ratio = Math.abs(outFloat[i] - expected[i]) / (FLOAT_UNIT * magnitude[i]);
      ratios[count] = ratio;
      strata[count++] = rowStratum[i / (d.queries() * WIDTH)];
      if (ratio > worst) {
        worst = ratio;
        worstAt = i;
      }
      if (ratio > CANCELLATION && outside++ < 8) {
        first.append(
            String.format(
                " [%d out=%s outFloat=%s fp64=%.9g A=%.4g ratio=%.1f]",
                i, out[i], outFloat[i], expected[i], magnitude[i], ratio));
      }
    }
    log.info(
        "PRISM-HIPATTN-OUT case={} rounded-once mismatches={} fp32 worst={} u·A (element {} |o|={} A={})"
            + " outside c={}: {}{}",
        shape,
        unrounded,
        String.format("%.2f", worst),
        worstAt,
        String.format("%.3g", Math.abs(expected[worstAt])),
        String.format("%.3g", magnitude[worstAt]),
        CANCELLATION,
        outside,
        first);
    log.info("PRISM-HIPATTN-OUT-Q case={} {}", shape, quantiles(Arrays.copyOf(ratios, count)));
    StringBuilder line = new StringBuilder();
    for (int k = 0; k < 3 + (d.keys() + 15) / 16; k++) {
      double[] part = new double[count];
      int n = 0;
      for (int i = 0; i < count; i++) {
        if (strata[i] == k) {
          part[n++] = ratios[i];
        }
      }
      if (n > 0) {
        double[] sorted = Arrays.copyOf(part, n);
        Arrays.sort(sorted);
        line.append(String.format(" %s:%d/%.2f/%.2f", stratumName(k), n, sorted[(int) (n * 0.99)], sorted[n - 1]));
      }
    }
    log.info("PRISM-HIPATTN-OUT-STRATA case={} n/p99/max{}", shape, line);
    checks.assertEquals(unrounded, 0, shape + " OUT is not OUT_FLOAT rounded once to BF16");
    checks.assertEquals(outside, 0, shape + " FP32 output outside c·2^-24·sum p|v| of FP64:" + first);
  }

  /** 分位を分けて出すための行の種類。穴の行は 0、抜けたタイルの行は 1、ほかは有効なキーを含むキーのタイルの数 B に 2 を足した値（有効なキーが無い行は 2）。 */
  private static int[] rowStrata(Inputs in) {
    RelationAttention.Dimensions d = in.d();
    int[] strata = new int[ROWS];
    for (int r = 0; r < ROWS; r++) {
      int blocks = 0;
      for (int kt = 0; 16 * kt < d.keys(); kt++) {
        boolean present = false;
        for (int j = 16 * kt; j < Math.min(16 * kt + 16, d.keys()); j++) {
          present |= in.mask()[r * d.keys() + j] != 0f;
        }
        blocks += present ? 1 : 0;
      }
      strata[r] =
          switch (r) {
            case HOLE_ROW -> 0;
            case GAP_ROW -> 1;
            default -> 2 + blocks;
          };
    }
    return strata;
  }

  private static String stratumName(int stratum) {
    return switch (stratum) {
      case 0 -> "hole";
      case 1 -> "gap";
      default -> "B" + (stratum - 2);
    };
  }

  /**
   * 比の分位（p50・p99・p99.9・p99.99・最大）を 1 行の文字列にする。分位 q の値は、並べ替えた列の (int)(n·q) 番目の要素。
   *
   * @param ratios 比の列。この配列そのものを並べ替える
   */
  private static String quantiles(double[] ratios) {
    int n = ratios.length;
    if (n == 0) {
      return "n=0";
    }
    Arrays.sort(ratios);
    return String.format(
        "n=%d p50=%.3f p99=%.3f p99.9=%.3f p99.99=%.3f max=%.3f",
        n, ratios[n / 2], ratios[(int) (n * 0.99)], ratios[(int) (n * 0.999)], ratios[(int) (n * 0.9999)], ratios[n - 1]);
  }

  /** BF16 の出力のうち、FP64 の参照を BF16 に丸めた値（FP64 → FP32 → BF16 の最近接偶数）とビットまで同じ要素の割合。 */
  private static double bfloat16Share(float[] values, double[] expected) {
    int same = 0;
    for (int i = 0; i < values.length; i++) {
      same += Float.floatToRawIntBits(values[i]) == Float.floatToRawIntBits(bfloat16(expected[i])) ? 1 : 0;
    }
    return same / (double) values.length;
  }

  /**
   * BF16 の勾配 {@code output} を要素ごとに検査する。g は FP32 の和 g₃₂ を BF16 に最近接偶数で 1 回丸めた値なので、|g − g₃₂| は g の
   * BF16 の半 ulp 以下になる。それを超えた分 e = |g − g₆₄| − ½ulp(g) は、FP32 の和の誤差の下限になる。要素は次のどちらかを満たせば
   * 幅に収まる。
   *
   * <ul>
   *   <li>1 ulp と rms·slack の幅: |g − g₆₄| ≤ ulp(g₆₄) + slack·rms(g₆₄)。桁落ちの無い要素はこの幅で見る。
   *   <li>桁落ちの幅: e ≤ c·2⁻²⁴·M（M は {@link Reference} の勾配の尺度、c = {@value #CANCELLATION}）。dS の行の和は 0 なので、
   *       dq = scale·Σ_j dS_ij k_j は必ず桁落ちする。|g| ≪ M の要素では、FP32 でも 1 つ目の幅を超えうる。
   * </ul>
   *
   * 比 e/(2⁻²⁴·M) の分位と、それぞれの幅を外れた要素の数をログに出す。両方を外れた要素は、検査に記録する前に全て数え、先頭の幾つかを
   * M と一緒に出す。
   *
   * @return FP64 の参照を BF16 に丸めた値とビットまで同じ要素の割合
   */
  private static double assertGradientElements(
      SoftAssert checks, Case shape, Output output, Map<Output, float[]> actual, Reference fp64, double slack) {
    float[] values = actual.get(output);
    double[] expected = fp64.values().get(output);
    double[] magnitude = fp64.magnitudes().get(output);
    double allowance = slack * rms(expected);
    double[] ratios = new double[values.length];
    int count = 0;
    int ulpOutside = 0;
    int cancellationOutside = 0;
    int outside = 0;
    StringBuilder first = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      double error = Math.abs(values[i] - expected[i]);
      double excess = Math.max(0, error - bfloat16Ulp(values[i]) / 2);
      double ratio = magnitude[i] == 0 ? (excess == 0 ? 0 : Double.POSITIVE_INFINITY) : excess / (FLOAT_UNIT * magnitude[i]);
      if (magnitude[i] > 0) {
        ratios[count++] = ratio;
      }
      boolean ulpRule = error <= bfloat16Ulp(expected[i]) + allowance;
      boolean cancellationRule = ratio <= CANCELLATION;
      ulpOutside += ulpRule ? 0 : 1;
      cancellationOutside += cancellationRule ? 0 : 1;
      if (!ulpRule && !cancellationRule && outside++ < 8) {
        first.append(String.format(" [%d %s fp64=%.9g M=%.4g ratio=%.1f]", i, values[i], expected[i], magnitude[i], ratio));
      }
    }
    log.info(
        "PRISM-HIPATTN-GRAD case={} {} {} outside 1ulp+rms={} outside c={}: {} outside both={}{}",
        shape,
        output,
        quantiles(Arrays.copyOf(ratios, count)),
        ulpOutside,
        CANCELLATION,
        cancellationOutside,
        outside,
        first);
    checks.assertEquals(
        outside, 0, shape + " " + output + " elements outside both 1 ulp + rms·slack and c·2^-24·M:" + first);
    return bfloat16Share(values, expected);
  }

  /**
   * 2 つの実行 {@code first}・{@code second} の出力 {@code outputs} を、実際の位置（クエリは先頭の {@code queries} 個、キーは先頭の
   * {@code keys} 個）で比べる。全てビットまで同じなら PRISM-HIPATTN-&lt;kind&gt; の行をログに出す。違う出力があれば、出力ごとに -DIFF の
   * 行を出し、{@code failures} に足す。
   */
  private static void compareRealPositions(
      String kind,
      String label,
      StringBuilder failures,
      Placed first,
      Placed second,
      int queries,
      int keys,
      Output... outputs) {
    boolean equal = true;
    for (Output output : outputs) {
      float[] want = realPositions(output, first.outputs().get(output), first.d(), first.rows(), queries, keys);
      float[] got = realPositions(output, second.outputs().get(output), second.d(), second.rows(), queries, keys);
      if (!bitsEqual(want, got)) {
        String differences = differences(output, want, got, queries, keys, first.d().tableRows());
        log.info("PRISM-HIPATTN-{}-DIFF {} {} {}", kind, label, output, differences);
        failures.append(String.format("%n%s %s differs: %s", label, output, differences));
        equal = false;
      }
    }
    if (equal) {
      log.info("PRISM-HIPATTN-{} {} bits equal", kind, label);
    }
  }

  /**
   * 実際の位置での 2 つの出力の違いを文字列にする。違う要素の数、そのうち ±0 だけが違う数（kv は dk・dv に分けた数も）、先頭の 8 個
   * （添字・位置・2 つの値・ビット）を書く。
   */
  private static String differences(Output output, float[] want, float[] got, int queries, int keys, int tableRows) {
    int differing = 0;
    int signedZeros = 0;
    int keyPart = 0;
    StringBuilder first = new StringBuilder();
    for (int i = 0; i < Math.min(want.length, got.length); i++) {
      int a = Float.floatToRawIntBits(want[i]);
      int b = Float.floatToRawIntBits(got[i]);
      if (a != b) {
        signedZeros += want[i] == 0f && got[i] == 0f ? 1 : 0;
        keyPart += output == Output.GRAD_KEY_VALUE && i % (2 * WIDTH) < WIDTH ? 1 : 0;
        if (differing++ < 8) {
          String at = position(output, i, queries, keys, tableRows);
          first.append(String.format(" [%d %s %s/%08x vs %s/%08x]", i, at, want[i], a, got[i], b));
        }
      }
    }
    String split = output == Output.GRAD_KEY_VALUE ? String.format(" dK=%d dV=%d", keyPart, differing - keyPart) : "";
    return String.format(
        "lengths %d/%d differing=%d signed-zero-only=%d%s%s",
        want.length, got.length, differing, signedZeros, split, first);
  }

  /** {@link #realPositions} の並びの {@code index} 番目が指す位置を文字列にする（並べた行の番号 n と、クエリかキー・ヘッド・次元。ρ の勾配はヘッドと表の行）。 */
  private static String position(Output output, int index, int queries, int keys, int tableRows) {
    return switch (output) {
      case OUT, OUT_FLOAT, GRAD_QUERY -> {
        int at = index % (queries * WIDTH);
        yield String.format(
            "n=%d query=%d head=%d feature=%d",
            index / (queries * WIDTH), at / WIDTH, at % WIDTH / FEATURES, at % FEATURES);
      }
      case LOG_SUM_EXP -> {
        int at = index % (HEADS * queries);
        yield String.format("n=%d head=%d query=%d", index / (HEADS * queries), at / queries, at % queries);
      }
      case GRAD_KEY_VALUE -> {
        int at = index % (keys * 2 * WIDTH);
        int column = at % (2 * WIDTH);
        yield String.format(
            "n=%d key=%d %s head=%d feature=%d",
            index / (keys * 2 * WIDTH),
            at / (2 * WIDTH),
            column < WIDTH ? "dK" : "dV",
            column % WIDTH / FEATURES,
            column % FEATURES);
      }
      case GRAD_TABLE -> {
        int at = index % (HEADS * tableRows);
        yield String.format("n=%d head=%d entry=%d", index / (HEADS * tableRows), at / tableRows, at % tableRows);
      }
    };
  }

  private static boolean bitsEqual(float[] first, float[] second) {
    if (first.length != second.length) {
      return false;
    }
    for (int i = 0; i < first.length; i++) {
      if (Float.floatToRawIntBits(first[i]) != Float.floatToRawIntBits(second[i])) {
        return false;
      }
    }
    return true;
  }

  /** 出力ごとの CRC32 をログに出す。別のプロセスで実行したログと比べるために使う。 */
  private static void printChecksums(Case shape, Map<Output, float[]> outputs) {
    StringBuilder line = new StringBuilder("PRISM-HIPATTN-CRC case=").append(shape);
    for (Output output : Output.values()) {
      float[] values = outputs.get(output);
      ByteBuffer bytes = ByteBuffer.allocate(4 * values.length).order(ByteOrder.LITTLE_ENDIAN);
      for (float value : values) {
        bytes.putInt(Float.floatToRawIntBits(value));
      }
      CRC32 crc = new CRC32();
      crc.update(bytes.array());
      line.append(' ').append(output).append('=').append(String.format("%08x", crc.getValue()));
    }
    log.info(line.toString());
  }

  /**
   * 出力 {@code output} から、行 {@code rows} の実際の位置の値を決まった順に並べて返す。実際の位置は、クエリなら先頭の {@code queries}
   * 個、キーなら先頭の {@code keys} 個、ρ の勾配なら行 × ヘッドの部分和の全て。
   */
  private static float[] realPositions(
      Output output, float[] values, RelationAttention.Dimensions d, int[] rows, int queries, int keys) {
    int length =
        switch (output) {
          case OUT, OUT_FLOAT, GRAD_QUERY -> queries * WIDTH;
          case LOG_SUM_EXP -> HEADS * queries;
          case GRAD_KEY_VALUE -> keys * 2 * WIDTH;
          case GRAD_TABLE -> HEADS * d.tableRows();
        };
    float[] real = new float[rows.length * length];
    for (int n = 0; n < rows.length; n++) {
      int r = rows[n];
      switch (output) {
        case OUT, OUT_FLOAT, GRAD_QUERY -> System.arraycopy(values, r * d.queries() * WIDTH, real, n * length, length);
        case LOG_SUM_EXP -> {
          for (int h = 0; h < HEADS; h++) {
            System.arraycopy(values, (r * HEADS + h) * d.queries(), real, n * length + h * queries, queries);
          }
        }
        case GRAD_KEY_VALUE -> System.arraycopy(values, r * d.keys() * 2 * WIDTH, real, n * length, length);
        case GRAD_TABLE -> System.arraycopy(values, r * HEADS * d.tableRows(), real, n * length, length);
      }
    }
    return real;
  }

  /** 行 0〜n − 1 を順に並べた配列。 */
  private static int[] identity(int n) {
    int[] rows = new int[n];
    Arrays.setAll(rows, i -> i);
    return rows;
  }

  /** 行 × ヘッドごとの ρ の勾配 [rows·heads, tableRows] を行で足して、[tableRows, heads] にする（FP64 で足す）。 */
  private static float[] rowSum(float[] gradTable, RelationAttention.Dimensions d) {
    double[] sum = new double[d.tableRows() * HEADS];
    for (int r = 0; r < d.rows(); r++) {
      for (int h = 0; h < HEADS; h++) {
        for (int e = 0; e < d.tableRows(); e++) {
          sum[e * HEADS + h] += gradTable[(r * HEADS + h) * d.tableRows() + e];
        }
      }
    }
    float[] result = new float[sum.length];
    for (int i = 0; i < sum.length; i++) {
      result[i] = (float) sum[i];
    }
    return result;
  }

  private static NDArray bfloat16(NDManager manager, float[] values, Shape shape) {
    return manager.create(values, shape).toType(DataType.BFLOAT16, false);
  }

  private static NDArray codes(NDManager manager, Inputs in) {
    RelationAttention.Dimensions d = in.d();
    return manager
        .create(in.codes(), new Shape(d.rows(), d.codeQueries(), d.keys(), WORDS))
        .toType(DataType.INT16, false);
  }

  private static float[] widen(NDArray array) {
    return array.toType(DataType.FLOAT32, false).toFloatArray();
  }

  private static float[] gaussian(Random random, int size, float scale) {
    float[] values = new float[size];
    for (int i = 0; i < size; i++) {
      values[i] = (float) random.nextGaussian() * scale;
    }
    return values;
  }

  /** BF16 で表せる正規乱数（FP32 の正規乱数を BF16 に丸めた値）。 */
  private static float[] bfloat16Gaussian(Random random, int size, float scale) {
    float[] values = gaussian(random, size, scale);
    for (int i = 0; i < size; i++) {
      values[i] = bfloat16(values[i]);
    }
    return values;
  }

  /** FP32 に丸めてから BF16 に最近接偶数で丸めた値（c10::BFloat16 と同じ）。 */
  private static float bfloat16(double value) {
    int bits = Float.floatToRawIntBits((float) value);
    int rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16;
    return Float.intBitsToFloat(rounded << 16);
  }

  private static double bfloat16Ulp(double value) {
    return Math.ulp((float) Math.abs(value)) * 65536.0;
  }

  private static double rms(double[] values) {
    double sum = 0;
    for (double value : values) {
      sum += value * value;
    }
    return Math.sqrt(sum / Math.max(values.length, 1));
  }

  private static double relative(float[] expected, float[] actual) {
    double error = 0;
    double norm = 0;
    for (int i = 0; i < expected.length; i++) {
      double d = (double) expected[i] - actual[i];
      error += d * d;
      norm += (double) expected[i] * expected[i];
    }
    return Math.sqrt(error / Math.max(norm, 1e-30));
  }

  private static double relative(double[] expected, float[] actual) {
    double error = 0;
    double norm = 0;
    for (int i = 0; i < expected.length; i++) {
      double d = expected[i] - actual[i];
      error += d * d;
      norm += expected[i] * expected[i];
    }
    return Math.sqrt(error / Math.max(norm, 1e-30));
  }

  /** 要素ごとの差の最大を、参照の絶対値の最大で割った比。 */
  private static double worst(double[] expected, float[] actual) {
    double largest = 0;
    double error = 0;
    for (int i = 0; i < expected.length; i++) {
      largest = Math.max(largest, Math.abs(expected[i]));
      error = Math.max(error, Math.abs(expected[i] - actual[i]));
    }
    return error / Math.max(largest, 1e-30);
  }
}
