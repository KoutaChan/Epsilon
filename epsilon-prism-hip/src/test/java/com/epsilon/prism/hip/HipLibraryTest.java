package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.prism.hip.HipLibrary.BuildInfo;
import com.epsilon.prism.hip.HipLibrary.RuntimeInfo;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class HipLibraryTest {

  private static final Logger log = LoggerFactory.getLogger(HipLibraryTest.class);

  private static final Path TORCH = Path.of("/djl/pytorch/2.11.0-rocm7.2-linux-x86_64");
  private static final Path HIP = TORCH.resolve("libamdhip64.so");
  private static final BuildInfo BUILD = new BuildInfo("2.11.0", "rocm7.2", "gfx1100");

  /** 読み込みの照らし: 揃った事実は通し、別の HIP の実行系・別の libtorch・版の違い・頭の大きさの違いは英語の例外で止める。 */
  @Test
  public void verifyRejectsMismatchedRuntimes() {
    HipLibrary.verify(BUILD, runtime(List.of(HIP), TORCH, 70253211, HipPack.HEADER_INTS), "2.11.0", TORCH, false);

    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD,
                runtime(
                    List.of(HIP, Path.of("/opt/rocm-7.2.2/lib/libamdhip64.so.7.2.70202")),
                    TORCH,
                    70253211,
                    HipPack.HEADER_INTS),
                "2.11.0",
                TORCH,
                false));
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD, runtime(List.of(HIP), Path.of("/elsewhere"), 70253211, HipPack.HEADER_INTS), "2.11.0", TORCH, false));
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(BUILD, runtime(List.of(HIP), TORCH, 70253211, HipPack.HEADER_INTS), "2.10.0", TORCH, false));
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(BUILD, runtime(List.of(HIP), TORCH, 100000000, HipPack.HEADER_INTS), "2.11.0", TORCH, false));
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD, runtime(List.of(HIP), TORCH, 70253211, HipPack.HEADER_INTS + 1), "2.11.0", TORCH, false));
  }

  /** WeightPack の頭は native の並び（magic・int の数・入力の約束・区画の幅・役割・重みの位置・埋め込み）で書かれる。 */
  @Test
  public void headerEncodesInNativeOrder() {
    HipPack.Header header =
        TestPack.header()
            .weight(HipWeight.QKV, 1, 4096)
            .role(3, HipComposition.TURN_GATE, 2)
            .embedding(
                HipEmbedding.Slot.TILE,
                new HipEmbedding(
                    HipSection.TILE_CATEGORY,
                    null,
                    null,
                    List.of(new HipEmbedding.Table(8192, 700, new int[] {0, 1}, new int[] {1, 35})),
                    List.of(new HipEmbedding.Dense(HipEmbedding.Source.FLAG, 2)),
                    16384,
                    20480));
    ByteBuffer pack = TestPack.direct(HipPack.HEADER_BYTES);
    header.write(pack);
    ByteBuffer ints = pack.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    int schemaAt = 2;
    int sectionsAt = schemaAt + HipSchema.values().length;
    int rolesAt = sectionsAt + HipSection.values().length;
    int gatesAt = rolesAt + HipPack.MAX_ROLES;
    int weightsAt = gatesAt + HipPack.MAX_ROLES;
    int embeddingsAt = weightsAt + HipWeight.values().length * HipShape.MAX_DEPTH;
    assertEquals(ints.getInt(0), HipPack.MAGIC);
    assertEquals(ints.getInt(4), HipPack.HEADER_INTS);
    assertEquals(ints.getInt((schemaAt + HipSchema.TILES.ordinal()) * 4), 34);
    assertEquals(ints.getInt((sectionsAt + HipSection.TRANSITION_CODE.ordinal()) * 4), TestPack.FIXED);
    assertEquals(ints.getInt((rolesAt + 3) * 4), HipComposition.TURN_GATE.ordinal());
    assertEquals(ints.getInt((gatesAt + 3) * 4), 2);
    assertEquals(ints.getInt((weightsAt + HipWeight.QKV.ordinal() * HipShape.MAX_DEPTH + 1) * 4), 4096);
    assertEquals(ints.getInt((weightsAt + HipWeight.QKV.ordinal() * HipShape.MAX_DEPTH) * 4), -1);
    int tile = embeddingsAt + HipEmbedding.Slot.TILE.ordinal() * HipEmbedding.INTS;
    assertEquals(ints.getInt(tile * 4), HipSection.TILE_CATEGORY.ordinal());
    assertEquals(ints.getInt((tile + 1) * 4), -1);
    assertEquals(ints.getInt((tile + 3) * 4), 1);
    assertEquals(ints.getInt((tile + 4) * 4), 1);
    assertEquals(ints.getInt((tile + 8) * 4), 8192);
    assertEquals(ints.getInt((tile + 9) * 4), 700);
    assertEquals(ints.getInt((tile + 11 + HipEmbedding.MAX_MEMBERS + 1) * 4), 35);
    assertEquals(ints.getInt((tile + 8 + HipEmbedding.MAX_TABLES * HipEmbedding.TABLE_INTS) * 4), 2);
    assertTrue(HipPack.HEADER_BYTES >= HipPack.HEADER_INTS * 4 && HipPack.HEADER_BYTES % HipPack.ALIGNMENT == 0);
  }

  /** サーバの CPU（GPU を隠す）で、DJL の ROCm の libtorch と同じプロセスに読み込め、HIP の実行系が 1 つに揃う。 */
  @Test(groups = "native")
  public void loadsBesideDjl() {
    if (HipLibrary.class.getResource("/native/linux-x86_64/libprism_hip.so") == null) {
      throw new SkipException("libprism_hip.so is built only with -Pnative-hip");
    }
    HipLibrary library = HipLibrary.load();
    RuntimeInfo runtime = library.runtime();
    log.info(
        "libprism_hip.so: HIP runtime {} at {}, built with {}; libtorch {}; header {} ints",
        runtime.hipRuntime(), runtime.hipLibrary(), runtime.hipCompiled(), runtime.torchLoaded(), runtime.headerInts());
    assertEquals(runtime.hipLoaded(), List.of(runtime.hipLibrary()));
    assertTrue(runtime.torchLoaded().stream().anyMatch(path -> path.endsWith("libtorch_hip.so")));
  }

  private static RuntimeInfo runtime(List<Path> hipLoaded, Path torchDirectory, int hipRuntime, int headerInts) {
    return new RuntimeInfo(
        70253211,
        hipRuntime,
        "",
        hipLoaded.get(0),
        hipLoaded,
        List.of(torchDirectory.resolve("libc10_hip.so"), torchDirectory.resolve("libtorch_hip.so")),
        true,
        false,
        HipKernel.values().length,
        HipRegion.values().length,
        headerInts,
        "gfx1100");
  }
}
