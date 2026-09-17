package com.epsilon.runtime;

import static org.testng.Assert.*;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.annotations.Test;

/** 計測対象の入れ子、ネイティブ実行記録の取得、保存失敗後の再開、実経過時間だけの計測を検証する。 */
public class InferenceProfileTest {
  @Test
  public void nestedAttachmentRestoresSelectionAndCapturesOnlyOnce() {
    var outer =
        new InferenceProfile("host.stage", Path.of("unused-stage.json"), Device.cpu(), true);
    var inner =
        new InferenceProfile("host.compose", Path.of("unused-compose.json"), Device.cpu(), true);
    assertNull(InferenceProfile.section("host.stage"));
    try (var attached = InferenceProfile.attach(outer)) {
      try (var nested = InferenceProfile.attach(inner)) {
        assertNull(InferenceProfile.section("host.stage"));
        try (var scope = InferenceProfile.section("host.compose")) {
          assertNotNull(scope);
        }
      }
      try (var scope = InferenceProfile.section("host.stage")) {
        assertNotNull(scope);
      }
      assertNull(InferenceProfile.section("host.stage"));
    }
    assertNull(InferenceProfile.section("host.compose"));
    assertNull(outer.result().traceFile());
    assertNull(inner.result().traceFile());
    assertTrue(outer.result().enqueueNanos() >= 0);
    assertEquals(outer.result().completedWallNanos(), outer.result().enqueueNanos());
  }

  @Test(groups = "native")
  public void nativeTraceCapturesTheSelectedOperationAndCanRestartAfterWriteFailure()
      throws Exception {
    Path directory = Files.createTempDirectory("epsilon-inference-profile-");
    Path trace = directory.resolve("add.json");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      var input = manager.create(new float[] {1, 2, 3});
      var failing = new InferenceProfile("model.add", directory, Device.cpu(), true);
      assertThrows(
          RuntimeException.class,
          () -> {
            try (var attached = InferenceProfile.attach(failing);
                var scope = InferenceProfile.section("model.add")) {
              input.add(1);
            }
          });
      assertNull(failing.result());
      var profile = new InferenceProfile("model.add", trace, Device.cpu(), true);
      try (var attached = InferenceProfile.attach(profile);
          var scope = InferenceProfile.section("model.add")) {
        input.add(2);
      }
      assertTrue(Files.readString(trace).contains("aten::add"));
      assertEquals(profile.result().traceFile(), trace.toString());
      assertTrue(profile.result().completedWallNanos() >= profile.result().enqueueNanos());
    } finally {
      Files.deleteIfExists(trace);
      Files.delete(directory);
    }
  }

  @Test(groups = "native")
  public void wallOnlyCapturesTheOperationWithoutWritingAnOperatorTrace() throws Exception {
    Path directory = Files.createTempDirectory("epsilon-inference-wall-");
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      var input = manager.create(new float[] {1, 2, 3});
      // ディレクトリは実行記録の出力先に使えない。実経過時間のみならネイティブ側のプロファイラーを起動しない。
      var profile = new InferenceProfile("model.add", directory, Device.cpu(), false);
      try (var attached = InferenceProfile.attach(profile);
          var scope = InferenceProfile.section("model.add")) {
        input.addi(2);
      }
      assertEquals(input.toFloatArray(), new float[] {3, 4, 5});
      assertNotNull(profile.result());
      assertNull(profile.result().traceFile());
      assertTrue(profile.result().enqueueNanos() >= 0);
      assertEquals(profile.result().completedWallNanos(), profile.result().enqueueNanos());
    } finally {
      Files.delete(directory);
    }
  }
}
