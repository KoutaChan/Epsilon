package com.epsilon.agent.checkpoint;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.expectThrows;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.mahjong.rules.Variant;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.stream.Stream;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * チェックポイントのディレクトリの書き込みと読み込みを、ネイティブを使わないブロックで確かめる。既にあるディレクトリには書かず、中身も
 * 変えない。書き込みや最後の rename が失敗しても、&lt;dir&gt;.tmp や書きかけのディレクトリは残らない。PRISM の model.json の
 * ヘッダーはそのまま読めて、書き直しても同じ文字列になる。
 */
public class CheckpointFilesTest {

  public record Header(String architecture, Variant variant, int hidden) {}

  /** PRISM の model.json のヘッダーの形。 */
  public record PrismHeader(String architecture, String inputSchema, String headSchema, PrismShape shape) {}

  public record PrismShape(int hidden, int ffn, int gate, int depth) {}

  /** PRISM が書く model.json。 */
  private static final String PRISM_MODEL_JSON =
      """
      {
        "architecture": "epsilon-prism-v2",
        "inputSchema": "0123456789abcdef",
        "headSchema": "fedcba9876543210",
        "shape": {
          "hidden": 256,
          "ffn": 768,
          "gate": 64,
          "depth": 3
        }
      }""";

  /** 公開した PRISM の model.json。今の形に無い "frontier" のキーを持つ。 */
  private static final String PUBLISHED_PRISM_MODEL_JSON =
      """
      {
        "architecture": "epsilon-prism-v2",
        "inputSchema": "0123456789abcdef",
        "headSchema": "fedcba9876543210",
        "shape": {
          "hidden": 256,
          "ffn": 768,
          "gate": 64,
          "depth": 3,
          "frontier": false
        }
      }""";

  private static final Header HEADER = new Header("test-v1", Variant.TENHOU_4P, 7);
  private static final Header OTHER = new Header("test-v1", Variant.TENHOU_3P, 9);

  private Path root;

  @BeforeMethod
  public void createRoot() throws IOException {
    root = Files.createTempDirectory("checkpoint");
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRoot() throws IOException {
    try (Stream<Path> files = Files.walk(root)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(file);
      }
    }
  }

  @Test
  public void writesWeightsAndHeaderAndReadsThemBack() throws IOException {
    Path directory = root.resolve("00012");
    CheckpointFiles.write(directory, new BytesBlock(new byte[] {1, 2, 3}), HEADER);
    assertEquals(fileNames(directory), List.of("model.json", "weights.bin"));
    assertEquals(CheckpointFiles.readHeader(directory, Header.class), HEADER);
    BytesBlock loaded = new BytesBlock(new byte[3]);
    CheckpointFiles.loadParameters(directory, loaded, null);
    assertEquals(loaded.bytes, new byte[] {1, 2, 3});
    assertFalse(Files.exists(root.resolve("00012.tmp")));
  }

  @Test
  public void aLeftoverTemporaryIsReplaced() throws IOException {
    Path directory = root.resolve("00013");
    Path leftover = root.resolve("00013.tmp");
    Files.createDirectories(leftover);
    Files.writeString(leftover.resolve("weights.bin"), "stale");
    CheckpointFiles.write(directory, new BytesBlock(new byte[] {2, 2}), OTHER);
    assertEquals(fileNames(directory), List.of("model.json", "weights.bin"));
    assertEquals(Files.readAllBytes(directory.resolve("weights.bin")), new byte[] {2, 2});
    assertEquals(CheckpointFiles.readHeader(directory, Header.class), OTHER);
    assertFalse(Files.exists(leftover));
  }

  /** 既にあるディレクトリには、ブロックに書かせる前に FileAlreadyExistsException を投げる。失敗するブロックが投げる例外にはならない。 */
  @Test
  public void anExistingDirectoryIsRefusedBeforeWriting() throws IOException {
    Path directory = root.resolve("00014");
    CheckpointFiles.write(directory, new BytesBlock(new byte[] {5}), HEADER);
    String before = CheckpointFiles.sha256(directory);
    expectThrows(FileAlreadyExistsException.class, () -> CheckpointFiles.write(directory, new FailingBlock(), OTHER));
    assertEquals(CheckpointFiles.sha256(directory), before);
    assertFalse(Files.exists(root.resolve("00014.tmp")));
  }

  @Test
  public void aFailedWriteLeavesNoPartialDirectory() {
    Path directory = root.resolve("00015");
    IOException failure =
        expectThrows(IOException.class, () -> CheckpointFiles.write(directory, new FailingBlock(), HEADER));
    assertEquals(failure.getMessage(), FailingBlock.MESSAGE);
    assertFalse(Files.exists(directory));
    assertFalse(Files.exists(root.resolve("00015.tmp")));
  }

  /** 重みを書いている間に、ほかの書き込みが同じディレクトリを作ると、最後の rename が失敗し、そのディレクトリは変わらない。 */
  @Test
  public void aFailedRenameLeavesTheOtherDirectory() throws IOException {
    Path directory = root.resolve("00016");
    Path other = directory.resolve("weights.bin");
    BytesBlock racing =
        new BytesBlock(new byte[] {7}) {
          @Override
          public void saveParameters(DataOutputStream os) throws IOException {
            super.saveParameters(os);
            Files.createDirectories(directory);
            Files.writeString(other, "other");
          }
        };
    expectThrows(IOException.class, () -> CheckpointFiles.write(directory, racing, HEADER));
    assertEquals(fileNames(directory), List.of("weights.bin"));
    assertEquals(Files.readString(other), "other");
    assertFalse(Files.exists(root.resolve("00016.tmp")));
  }

  @Test
  public void anUninitializedBlockIsRejectedBeforeWriting() {
    Path directory = root.resolve("00017");
    expectThrows(IllegalArgumentException.class, () -> CheckpointFiles.write(directory, new SequentialBlock(), HEADER));
    assertFalse(Files.exists(directory));
    assertFalse(Files.exists(root.resolve("00017.tmp")));
  }

  @Test
  public void trailingWeightsAreRejected() throws IOException {
    Path directory = root.resolve("00018");
    CheckpointFiles.write(directory, new BytesBlock(new byte[] {1, 2}), HEADER);
    expectThrows(IOException.class, () -> CheckpointFiles.loadParameters(directory, new BytesBlock(new byte[1]), null));
  }

  @Test
  public void readsThePrismHeaderAndWritesTheSameText() throws IOException {
    Path saved = root.resolve("m0");
    Files.createDirectories(saved);
    Files.writeString(saved.resolve("model.json"), PRISM_MODEL_JSON);
    PrismHeader header = CheckpointFiles.readHeader(saved, PrismHeader.class);
    assertEquals(
        header,
        new PrismHeader("epsilon-prism-v2", "0123456789abcdef", "fedcba9876543210", new PrismShape(256, 768, 64, 3)));
    Path rewritten = root.resolve("m1");
    CheckpointFiles.write(rewritten, new BytesBlock(new byte[] {0}), header);
    assertEquals(Files.readString(rewritten.resolve("model.json")), PRISM_MODEL_JSON);
  }

  /** 公開した model.json の知らないキーは読み飛ばす。書き直すと今の形になる。 */
  @Test
  public void thePublishedPrismHeaderReadsAndWritesTheCurrentText() throws IOException {
    Path published = root.resolve("published");
    Files.createDirectories(published);
    Files.writeString(published.resolve("model.json"), PUBLISHED_PRISM_MODEL_JSON);
    PrismHeader header = CheckpointFiles.readHeader(published, PrismHeader.class);
    assertEquals(
        header,
        new PrismHeader("epsilon-prism-v2", "0123456789abcdef", "fedcba9876543210", new PrismShape(256, 768, 64, 3)));
    Path rewritten = root.resolve("current");
    CheckpointFiles.write(rewritten, new BytesBlock(new byte[] {0}), header);
    assertEquals(Files.readString(rewritten.resolve("model.json")), PRISM_MODEL_JSON);
  }

  @Test
  public void brokenHeadersAreRejected() throws IOException {
    Path directory = root.resolve("broken");
    Files.createDirectories(directory);
    expectThrows(IOException.class, () -> CheckpointFiles.readHeader(directory, Header.class));
    Files.writeString(directory.resolve("model.json"), "{\"hidden\": [");
    expectThrows(IOException.class, () -> CheckpointFiles.readHeader(directory, Header.class));
    Files.writeString(directory.resolve("model.json"), "null");
    expectThrows(IOException.class, () -> CheckpointFiles.readHeader(directory, Header.class));
  }

  /** SHA-256 は、通常のファイルをパスの順に並べ、ファイルごとに相対パス・0x00・大きさ・中身の順で足して求める。中身か名前が変われば値も変わる。 */
  @Test
  public void sha256FollowsTheDocumentedLayout() throws IOException {
    Path directory = root.resolve("digest");
    Files.createDirectories(directory.resolve("sub"));
    Files.writeString(directory.resolve("a.txt"), "x");
    Files.write(directory.resolve("sub").resolve("b.bin"), new byte[] {1, 2});
    MessageDigest expected = sha256();
    entry(expected, "a.txt", "x".getBytes(StandardCharsets.UTF_8));
    entry(expected, "sub/b.bin", new byte[] {1, 2});
    String digest = CheckpointFiles.sha256(directory);
    assertEquals(digest, HexFormat.of().formatHex(expected.digest()));
    Files.write(directory.resolve("sub").resolve("b.bin"), new byte[] {1, 3});
    String changed = CheckpointFiles.sha256(directory);
    assertNotEquals(changed, digest);
    Files.move(directory.resolve("a.txt"), directory.resolve("c.txt"));
    assertNotEquals(CheckpointFiles.sha256(directory), changed);
    Path empty = root.resolve("empty");
    Files.createDirectories(empty);
    expectThrows(IOException.class, () -> CheckpointFiles.sha256(empty));
  }

  private static void entry(MessageDigest digest, String name, byte[] content) {
    digest.update(name.getBytes(StandardCharsets.UTF_8));
    digest.update((byte) 0);
    digest.update(ByteBuffer.allocate(Long.BYTES).putLong(content.length).array());
    digest.update(content);
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException missing) {
      throw new IllegalStateException(missing);
    }
  }

  private static List<String> fileNames(Path directory) throws IOException {
    try (Stream<Path> files = Files.list(directory)) {
      return files.map(file -> file.getFileName().toString()).sorted().toList();
    }
  }

  /** 決まったバイト列を重みとして書き、読むときは同じ長さを読むブロック。 */
  private static class BytesBlock extends AbstractBlock {

    private final byte[] bytes;

    BytesBlock(byte[] bytes) {
      this.bytes = bytes;
    }

    @Override
    public boolean isInitialized() {
      return true;
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
      os.write(bytes);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
      is.readFully(bytes);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore store, NDList inputs, boolean training, PairList<String, Object> params) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      throw new UnsupportedOperationException();
    }
  }

  /** 重みを途中まで書いてから失敗するブロック。 */
  private static final class FailingBlock extends AbstractBlock {

    static final String MESSAGE = "No space left on device";

    @Override
    public boolean isInitialized() {
      return true;
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
      os.write(new byte[64]);
      os.flush();
      throw new IOException(MESSAGE);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore store, NDList inputs, boolean training, PairList<String, Object> params) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      throw new UnsupportedOperationException();
    }
  }
}
