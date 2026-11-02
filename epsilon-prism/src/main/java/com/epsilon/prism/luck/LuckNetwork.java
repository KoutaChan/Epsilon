package com.epsilon.prism.luck;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.prism.luck.LuckInputSchema.Token;
import com.epsilon.prism.model.Context;
import com.epsilon.prism.model.EncoderLayer;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.RmsNorm;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 運の網 Ω。打つ網とは別の網で、共有するのはコード（部品）だけ: 埋め込み（{@link LuckEmbedding}）→ 背骨の層
 * （{@link EncoderLayer}）× {@value #DEPTH} → 最後の RMSNorm → 結果の頭（{@link LuckHead}）。入力は全員の手牌とプールを見る 39 トークン（出来事の
 * トークンなし）で、全ての層で全てのトークンを問い合わせにする。出力は結果 37 通りごとの局末の Φ の 4 席分。
 *
 * <p>深さは {@value #DEPTH} に固定し、幅は設定（{@link #shape(int)}。256 か 384）。子の登録の順は保存の形式の一部: embedding、encoder0、
 * encoder1、finalNorm、head。白紙では頭の最後の層が 0 なので V ≡ 0（c ≡ 0）。
 */
public final class LuckNetwork extends AbstractBlock {

  /** model.json の網の形式。 */
  public static final String ARCHITECTURE = "epsilon-prism-v2-luck";

  /** 背骨の層の数（固定）。 */
  public static final int DEPTH = 2;

  private static final String WEIGHTS = "weights.bin";
  private static final String METADATA = "model.json";
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  /** model.json の中身。 */
  private record Metadata(String architecture, String inputSchema, PrismShape shape) {}

  private final PrismShape shape;
  private final LuckEmbedding embedding;
  private final EncoderLayer[] encoders;
  private final RmsNorm finalNorm;
  private final LuckHead head;

  /**
   * 網を組む（パラメーターは {@link #initialize(NDManager)} で作る）。
   *
   * @param shape 形（深さは {@value #DEPTH}）
   * @throws IllegalArgumentException 深さが {@value #DEPTH} でないとき
   */
  public LuckNetwork(PrismShape shape) {
    if (shape.depth() != DEPTH) {
      throw new IllegalArgumentException("the luck network has depth " + DEPTH + ": " + shape);
    }
    this.shape = shape;
    embedding = addChildBlock("embedding", new LuckEmbedding(shape));
    encoders = new EncoderLayer[DEPTH];
    for (int layer = 0; layer < DEPTH; layer++) {
      encoders[layer] = addChildBlock("encoder" + layer, new EncoderLayer(shape, layer));
    }
    finalNorm = addChildBlock("finalNorm", new RmsNorm(shape));
    head = addChildBlock("head", new LuckHead(shape));
  }

  /**
   * 幅 W の形を返す（F = 2W、頭の幅 G = W、深さ {@value #DEPTH}）。
   *
   * @param width 流れの幅 W（64 の倍数）
   * @return 形
   */
  public static PrismShape shape(int width) {
    return new PrismShape(width, 2 * width, width, DEPTH, false);
  }

  /**
   * 白紙の網を作る。初期値は種で決まる。
   *
   * @param shape 形
   * @param manager パラメーターの管理元
   * @param seed 初期値の種
   * @return 網
   */
  public static LuckNetwork blank(PrismShape shape, NDManager manager, long seed) {
    LuckNetwork network = new LuckNetwork(shape);
    synchronized (LuckNetwork.class) {
      manager.getEngine().setRandomSeed(Long.hashCode(seed));
      network.initialize(manager);
    }
    return network;
  }

  public PrismShape shape() {
    return shape;
  }

  /**
   * パラメーターを FP32 で作る（初期値は管理元のエンジンの乱数の種で決まる）。
   *
   * @param manager パラメーターの管理元
   */
  public void initialize(NDManager manager) {
    initialize(manager, DataType.FLOAT32);
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... ignored) {
    for (Block child : getChildren().values()) {
      child.initialize(manager, dataType);
    }
  }

  /**
   * 順伝播。autocast は開かずに呼ぶ（丸める所は {@link Precision} が決める）。
   *
   * @param store パラメーターの値の取り元
   * @param inputs 入力（作った配列は入力の管理元に付く）
   * @param precision 計算の型
   * @param training 学習の順伝播なら true
   * @return V [行, 37, 4]（FP32、4 席の和は 0。結果の軸は正準の結果の番号）
   */
  public NDArray forward(
      ParameterStore store, LuckInputs inputs, Precision precision, boolean training) {
    Context context = new Context(store, inputs.manager(), precision, training);
    int rows = inputs.rows();
    NDList parts = new NDList(Token.values().length);
    for (Token token : Token.values()) {
      parts.add(
          embedding.embed(context, inputs, token).reshape(rows, token.perRow(), shape.hidden()));
    }
    NDArray tokens = NDArrays.concat(parts, 1);
    NDArray mask = context.working().ones(new Shape(rows, Token.ROW_TOKENS), DataType.FLOAT32);
    for (EncoderLayer encoder : encoders) {
      tokens = encoder.forward(context, tokens, mask, inputs.codes(), Token.ROW_TOKENS);
    }
    return head.forward(context, finalNorm.apply(context, tokens, DEPTH - 1));
  }

  /**
   * 重み（weights.bin）と model.json（網の形式・入力の指紋・形）を書く。
   *
   * @param directory 書く先
   * @throws IOException 書けないとき
   */
  public void save(Path directory) throws IOException {
    Files.createDirectories(directory);
    try (DataOutputStream out =
        new DataOutputStream(
            new BufferedOutputStream(Files.newOutputStream(directory.resolve(WEIGHTS))))) {
      saveParameters(out);
    }
    Files.writeString(
        directory.resolve(METADATA),
        GSON.toJson(new Metadata(ARCHITECTURE, LuckInputSchema.fingerprint(), shape)));
  }

  /**
   * 保存した網を読む。網の形式と入力の指紋を照らす。
   *
   * @param directory 保存した所
   * @param manager パラメーターの管理元
   * @return 網
   * @throws IOException 読めないか、形式・指紋・重みが合わないとき
   */
  public static LuckNetwork load(Path directory, NDManager manager) throws IOException {
    Metadata metadata;
    try {
      metadata = GSON.fromJson(Files.readString(directory.resolve(METADATA)), Metadata.class);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid luck network metadata " + directory, invalid);
    }
    if (metadata == null
        || metadata.shape() == null
        || !ARCHITECTURE.equals(metadata.architecture())) {
      throw new IOException("Not a luck network: " + directory);
    }
    if (!LuckInputSchema.fingerprint().equals(metadata.inputSchema())) {
      throw new IOException("Luck network input schema mismatch " + directory);
    }
    PrismShape saved = metadata.shape();
    LuckNetwork network =
        new LuckNetwork(
            new PrismShape(saved.hidden(), saved.ffn(), saved.gate(), saved.depth(), false));
    network.initialize(manager);
    try (DataInputStream in =
        new DataInputStream(
            new BufferedInputStream(Files.newInputStream(directory.resolve(WEIGHTS))))) {
      network.loadParameters(manager, in);
      if (in.read() != -1) {
        throw new IOException("Trailing luck network weight data " + directory);
      }
    } catch (MalformedModelException invalid) {
      throw new IOException("Invalid luck network weights " + directory, invalid);
    }
    return network;
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore store, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException(
        "Use forward(ParameterStore, LuckInputs, Precision, boolean)");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    throw new UnsupportedOperationException("Shapes follow LuckInputs");
  }
}
