package com.epsilon.prism.kernel;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.kernel.KernelProvider.Cache;
import com.epsilon.prism.kernel.KernelProvider.Capabilities;
import com.epsilon.prism.kernel.KernelProvider.Capacity;
import com.epsilon.prism.kernel.KernelProvider.Operation;
import com.epsilon.prism.kernel.KernelProvider.Plan;
import com.epsilon.prism.kernel.KernelProvider.Request;
import com.epsilon.prism.kernel.KernelProvider.Use;
import com.epsilon.prism.kernel.ModelSchema.Arithmetic;
import com.epsilon.prism.kernel.ModelSchema.Axis;
import com.epsilon.prism.kernel.ModelSchema.Composition;
import com.epsilon.prism.kernel.ModelSchema.Element;
import com.epsilon.prism.kernel.ModelSchema.FeedForward;
import com.epsilon.prism.kernel.ModelSchema.Input;
import com.epsilon.prism.kernel.ModelSchema.Layer;
import com.epsilon.prism.kernel.ModelSchema.Mask;
import com.epsilon.prism.kernel.ModelSchema.Model;
import com.epsilon.prism.kernel.ModelSchema.Normalization;
import com.epsilon.prism.kernel.ModelSchema.Order;
import com.epsilon.prism.kernel.ModelSchema.Queries;
import com.epsilon.prism.kernel.ModelSchema.Role;
import com.epsilon.prism.kernel.ModelSchema.Source;
import com.epsilon.prism.kernel.ModelSchema.Stage;
import com.epsilon.prism.kernel.ModelSchema.Weight;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import org.testng.annotations.Test;

/**
 * モデルの宣言の約束を確かめる。Java で書いた宣言を C++ が同じ値で読み書きすること、読み込むライブラリの宣言の ABI と JNI の口の
 * バージョンを照らすこと、壊れた宣言を項目を読む前に拒むこと、収集のプロバイダーは宣言の要る演算を全て持つ物だけを選ぶことを見る。
 */
public class DeclarationContractTest {
  static Model model() {
    return model("fixture");
  }

  /** weight の名前だけで、Java と native のどちらが書いた宣言かを見分ける。 */
  static Model model(String weight) {
    return new Model(
        3,
        27,
        48,
        96,
        3,
        16,
        1e-5f,
        .25f,
        .01f,
        64,
        128,
        96,
        1,
        4,
        Element.BF16,
        Element.FP32,
        Element.FP32,
        Order.LEGAL_ACTION,
        0,
        31,
        0,
        List.of(new Input("ACTION_COLUMN", Axis.ACTION, Element.I32, 1, List.of())),
        List.of(
            new Layer(
                Queries.ALL_TOKENS,
                Queries.ALL_TOKENS,
                Mask.ALL_KEYS,
                Source.EMBEDDING,
                Normalization.PRE_ATTENTION_AND_PRE_FFN,
                FeedForward.SHARED,
                128,
                128)),
        List.of(),
        List.of(Stage.COMPOSE_VALUES),
        List.of(new Role("PASS", Composition.ZERO)),
        List.of(Arithmetic.VALUE_ACTION_ORDER),
        List.of(new Weight(weight, Element.FP32, List.of(1))));
  }

  @Test
  public void exchangesDeclarationsWithCpp() throws Exception {
    var exe =
        Path.of(
            "target",
            "native-tests",
            "bin",
            System.getProperty("os.name").startsWith("Windows")
                ? "declaration_round_trip.exe"
                : "declaration_round_trip");
    Path file = Files.createTempFile(Path.of("target"), "declaration-", ".bin");
    try {
      var wire = ModelDeclarations.encode(model());
      byte[] bytes = new byte[wire.remaining()];
      wire.get(bytes);
      Files.write(file, bytes);
      assertEquals(
          new ProcessBuilder(exe.toString(), "--check-java", file.toString())
              .inheritIO()
              .start()
              .waitFor(),
          0);
      bytes[4]--;
      Files.write(file, bytes);
      assertNotEquals(
          new ProcessBuilder(exe.toString(), "--check-java", file.toString())
              .inheritIO()
              .start()
              .waitFor(),
          0);
      Files.delete(file);
      assertEquals(
          new ProcessBuilder(exe.toString(), "--write", file.toString())
              .inheritIO()
              .start()
              .waitFor(),
          0);
      var nativeBytes = ByteBuffer.wrap(Files.readAllBytes(file));
      assertEquals(
          ModelDeclarations.decode(nativeBytes, nativeBytes.remaining(), ModelSchema.VERSION),
          model("native"));
    } finally {
      Files.deleteIfExists(file);
    }
  }

  @Test
  public void checksLoadedVersions() {
    String module = "epsilon-prism-kernel-common";
    int jni = NativeDeclarations.JNI_VERSION;
    NativeDeclarations.requireLibrary(module, ModelSchema.VERSION, jni);
    assertThrows(
        IllegalStateException.class,
        () -> NativeDeclarations.requireLibrary(module, ModelSchema.VERSION - 1, jni));
    assertThrows(
        IllegalStateException.class,
        () -> NativeDeclarations.requireLibrary(module, ModelSchema.VERSION, jni + 1));
  }

  @Test
  public void checksEnvelopeBeforeReadingFields() {
    var model = model();
    var wire = ModelDeclarations.encode(model);
    assertEquals(ModelDeclarations.decode(wire, wire.remaining(), ModelSchema.VERSION), model);
    int length = wire.remaining();
    for (int n : List.of(0, 3, length - 1))
      assertThrows(
          IllegalArgumentException.class,
          () -> ModelDeclarations.decode(wire.slice(0, n), n, ModelSchema.VERSION));
    assertThrows(
        IllegalArgumentException.class,
        () -> ModelDeclarations.decode(wire, length, ModelSchema.VERSION + 1));
    for (int at : List.of(0, 4, 8, 12)) {
      var copy =
          ByteBuffer.allocate(length).order(ByteOrder.LITTLE_ENDIAN).put(wire.duplicate()).flip();
      copy.putInt(at, -1);
      assertThrows(
          IllegalArgumentException.class,
          () -> ModelDeclarations.decode(copy, length, ModelSchema.VERSION));
    }
    var extra =
        ByteBuffer.allocate(length + 4)
            .order(ByteOrder.LITTLE_ENDIAN)
            .put(wire.duplicate())
            .putInt(99)
            .flip();
    int bodyLength = 2 * Integer.BYTES;
    extra.putInt(bodyLength, extra.getInt(bodyLength) + 4);
    assertThrows(
        IllegalArgumentException.class,
        () -> ModelDeclarations.decode(extra, extra.remaining(), ModelSchema.VERSION));
  }

  @Test
  public void rejectsSectionsWithWrongAxesOrTypes() {
    var wire = ModelDeclarations.encode(model());
    int n = wire.remaining();
    byte[] bytes = new byte[n];
    wire.get(bytes);
    int name =
        new String(bytes, StandardCharsets.ISO_8859_1).indexOf("ACTION_COLUMN")
            + "ACTION_COLUMN".length();
    assertTrue(name > "ACTION_COLUMN".length());
    for (int offset : List.of(name, name + Integer.BYTES)) {
      var damaged = ByteBuffer.wrap(bytes.clone()).order(ByteOrder.LITTLE_ENDIAN);
      damaged.putInt(offset, 1);
      assertThrows(
          IllegalArgumentException.class,
          () -> ModelDeclarations.decode(damaged, n, ModelSchema.VERSION));
    }
  }

  @Test
  public void nativeCollectionNeedsEveryDeclaredOperation() {
    var req =
        new Request(
            model(),
            Backend.values()[0],
            0,
            Element.BF16,
            Use.COLLECTION,
            new Capacity(2, 8, 3, 4, 32),
            Cache.OFF);
    record Provider(int abiVersion, Set<Operation> ops) implements KernelProvider {
      public Backend backend() {
        return Backend.values()[0];
      }

      public Capabilities capabilities(Request ignored) {
        return new Capabilities(ops, "fixture");
      }

      public Plan open(Request request, long stream) {
        throw new AssertionError("Discovery loaded a plan");
      }
    }
    var none =
        expectThrows(IllegalStateException.class, () -> KernelProviders.collection(req, List.of()));
    assertTrue(none.getMessage().contains("GPU 0"), none.getMessage());
    var partial = new Provider(ModelSchema.VERSION, Set.of(Operation.FULL_SUBMISSION));
    var lacking =
        expectThrows(
            IllegalStateException.class, () -> KernelProviders.collection(req, List.of(partial)));
    assertTrue(lacking.getMessage().contains("fixture"), lacking.getMessage());
    var complete =
        new Provider(
            ModelSchema.VERSION,
            Set.of(
                Operation.FULL_SUBMISSION,
                Operation.RELATION_CODES,
                Operation.TOKEN_EMBEDDING,
                Operation.RELATION_ATTENTION,
                Operation.FEED_FORWARD,
                Operation.COMPOSE_VALUES));
    assertSame(KernelProviders.collection(req, List.of(partial, complete)), complete);
    var abi =
        expectThrows(
            IllegalStateException.class,
            () ->
                KernelProviders.collection(
                    req, List.of(new Provider(ModelSchema.VERSION + 1, complete.ops()))));
    assertTrue(
        abi.getMessage().contains(Integer.toString(ModelSchema.VERSION + 1)), abi.getMessage());
    assertThrows(
        IllegalStateException.class,
        () -> KernelProviders.collection(req, List.of(complete, complete)));
  }
}
