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

  /**
   * 読み込んだライブラリの照合: そろった情報は通す。別の HIP のランタイム・別の libtorch・バージョンの違い・形の記述やヘッダーの大きさの
   * 違いは、英語のメッセージの例外で止める。
   */
  @Test
  public void verifyRejectsMismatchedRuntimes() {
    HipLibrary.verify(BUILD, runtime(List.of(HIP), TORCH, 70253211, HipPack.HEADER_INTS), "2.11.0", TORCH, false);
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD,
                runtime(List.of(HIP), TORCH, 70253211, HipArchitecture.INTS + 1, HipPack.HEADER_INTS),
                "2.11.0",
                TORCH,
                false));

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

  /**
   * WeightPack のヘッダーは native の並び（形の記述・入力の列の決まり・領域の幅・手の扱いごとの作り方とゲート・重みの位置・埋め込み）で
   * 書かれ、形の記述から始まる。重みの位置は、重みごとの引き方の項目の数だけ続く。
   */
  @Test
  public void headerEncodesInNativeOrder() {
    HipArchitecture architecture = TestPack.architecture(512, TestPack.FOUR_PLAYERS);
    HipPack.Header header =
        TestPack.header(architecture)
            .weight(HipWeight.QKV, 1, 4096)
            .weight(HipWeight.UP, HipWeight.ffnEntry(1, TokenGroup.ALL), 8192)
            .weight(HipWeight.UP, HipWeight.CANDIDATE_FFN, 12288)
            .role(3, HipComposition.TURN_GATE, 2)
            .embedding(
                HipEmbedding.Slot.TILE,
                new HipEmbedding(
                    HipSection.TILE_CATEGORY,
                    null,
                    null,
                    List.of(new HipEmbedding.Table(8192, 700, new int[] {0, 1}, new int[] {1, 35}, 24576)),
                    List.of(new HipEmbedding.Dense(HipEmbedding.Source.FLAG, 2)),
                    16384,
                    20480,
                    32768));
    ByteBuffer pack = TestPack.direct(HipPack.HEADER_BYTES);
    header.write(pack);
    ByteBuffer ints = pack.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    int architectureAt = 0;
    int schemaAt = architectureAt + HipArchitecture.INTS;
    int sectionsAt = schemaAt + HipSchema.values().length;
    int rolesAt = sectionsAt + HipSection.values().length;
    int gatesAt = rolesAt + HipArchitecture.MAX_ROLES;
    int weightsAt = gatesAt + HipArchitecture.MAX_ROLES;
    int entries = 0;
    for (HipWeight weight : HipWeight.values()) {
      entries += weight.index().entries();
    }
    int embeddingsAt = weightsAt + entries;
    assertEquals(entries, HipWeight.ENTRIES);
    assertEquals(embeddingsAt + HipEmbedding.Slot.values().length * HipEmbedding.INTS, HipPack.HEADER_INTS);
    int[] encoded = architecture.encode();
    for (int i = 0; i < encoded.length; i++) {
      assertEquals(ints.getInt((architectureAt + i) * 4), encoded[i], "architecture int " + i);
    }
    assertEquals(ints.getInt((schemaAt + HipSchema.RELATION_ROWS.ordinal()) * 4), 112);
    assertEquals(
        ints.getInt((sectionsAt + HipSection.TRANSITION_CODE.ordinal()) * 4), architecture.counts().fixed());
    assertEquals(ints.getInt((rolesAt + 3) * 4), HipComposition.TURN_GATE.ordinal());
    assertEquals(ints.getInt((gatesAt + 3) * 4), 2);
    assertEquals(ints.getInt((weightsAt + HipWeight.QKV.firstEntry() + 1) * 4), 4096);
    assertEquals(ints.getInt((weightsAt + HipWeight.QKV.firstEntry()) * 4), -1);
    assertEquals(ints.getInt((weightsAt + HipWeight.UP.firstEntry() + 3) * 4), 8192);
    assertEquals(ints.getInt((weightsAt + HipWeight.UP.firstEntry() + 9) * 4), 12288);
    int tile = embeddingsAt + HipEmbedding.Slot.TILE.ordinal() * HipEmbedding.INTS;
    assertEquals(ints.getInt(tile * 4), HipSection.TILE_CATEGORY.ordinal());
    assertEquals(ints.getInt((tile + 1) * 4), -1);
    assertEquals(ints.getInt((tile + 3) * 4), 1);
    assertEquals(ints.getInt((tile + 4) * 4), 1);
    assertEquals(ints.getInt((tile + 8) * 4), 8192);
    assertEquals(ints.getInt((tile + 9) * 4), 700);
    assertEquals(ints.getInt((tile + 11 + HipEmbedding.MAX_MEMBERS + 1) * 4), 35);
    assertEquals(ints.getInt((tile + 7) * 4), 32768);
    assertEquals(ints.getInt((tile + 11 + 2 * HipEmbedding.MAX_MEMBERS) * 4), 24576);
    assertEquals(ints.getInt((tile + 8 + HipEmbedding.MAX_TABLES * HipEmbedding.TABLE_INTS) * 4), 2);
    assertTrue(HipPack.HEADER_BYTES >= HipPack.HEADER_INTS * 4 && HipPack.HEADER_BYTES % HipPack.ALIGNMENT == 0);
  }

  /** DJL の ROCm の libtorch と同じプロセスに読み込め、HIP のランタイムが 1 つにそろう。 */
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
    return runtime(hipLoaded, torchDirectory, hipRuntime, HipArchitecture.INTS, headerInts);
  }

  private static RuntimeInfo runtime(
      List<Path> hipLoaded, Path torchDirectory, int hipRuntime, int architectureInts, int headerInts) {
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
        architectureInts,
        headerInts,
        "gfx1100");
  }
}
