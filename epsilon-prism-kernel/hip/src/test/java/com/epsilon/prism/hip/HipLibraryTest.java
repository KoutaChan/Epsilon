package com.epsilon.prism.hip;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.prism.hip.HipLibrary.BuildInfo;
import com.epsilon.prism.hip.HipLibrary.RuntimeInfo;
import com.epsilon.prism.kernel.CollectionArchitecture;
import com.epsilon.prism.kernel.PackedWeightLayout;
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
    HipLibrary.verify(
        BUILD,
        runtime(List.of(HIP), TORCH, 70253211, PackedWeightLayout.HEADER_INTS),
        "2.11.0",
        TORCH,
        false);
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD,
                runtime(
                    List.of(HIP),
                    TORCH,
                    70253211,
                    CollectionArchitecture.INTS + 1,
                    PackedWeightLayout.HEADER_INTS),
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
                    PackedWeightLayout.HEADER_INTS),
                "2.11.0",
                TORCH,
                false));
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD,
                runtime(
                    List.of(HIP), Path.of("/elsewhere"), 70253211, PackedWeightLayout.HEADER_INTS),
                "2.11.0",
                TORCH,
                false));
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD,
                runtime(List.of(HIP), TORCH, 70253211, PackedWeightLayout.HEADER_INTS),
                "2.10.0",
                TORCH,
                false));
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD,
                runtime(List.of(HIP), TORCH, 100000000, PackedWeightLayout.HEADER_INTS),
                "2.11.0",
                TORCH,
                false));
    assertThrows(
        IllegalStateException.class,
        () ->
            HipLibrary.verify(
                BUILD,
                runtime(List.of(HIP), TORCH, 70253211, PackedWeightLayout.HEADER_INTS + 1),
                "2.11.0",
                TORCH,
                false));
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
        runtime.hipRuntime(),
        runtime.hipLibrary(),
        runtime.hipCompiled(),
        runtime.torchLoaded(),
        runtime.headerInts());
    assertEquals(runtime.hipLoaded(), List.of(runtime.hipLibrary()));
    assertTrue(runtime.torchLoaded().stream().anyMatch(path -> path.endsWith("libtorch_hip.so")));
  }

  private static RuntimeInfo runtime(
      List<Path> hipLoaded, Path torchDirectory, int hipRuntime, int headerInts) {
    return runtime(hipLoaded, torchDirectory, hipRuntime, CollectionArchitecture.INTS, headerInts);
  }

  private static RuntimeInfo runtime(
      List<Path> hipLoaded,
      Path torchDirectory,
      int hipRuntime,
      int architectureInts,
      int headerInts) {
    return new RuntimeInfo(
        70253211,
        hipRuntime,
        "",
        hipLoaded.get(0),
        hipLoaded,
        List.of(torchDirectory.resolve("libc10_hip.so"), torchDirectory.resolve("libtorch_hip.so")),
        true,
        false,
        architectureInts,
        headerInts,
        "gfx1100");
  }
}
