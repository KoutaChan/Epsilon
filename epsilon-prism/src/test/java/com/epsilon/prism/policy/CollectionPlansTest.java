package com.epsilon.prism.policy;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;
import static com.epsilon.prism.input.InputSchema.THREE_PLAYER;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import ai.djl.Device;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.backend.KernelInputs;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.kernel.CollectionKernels;
import com.epsilon.prism.kernel.KernelProvider;
import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.kernel.KernelProvider.Cache;
import com.epsilon.prism.kernel.ModelSchema;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.testng.annotations.Test;

/**
 * 専用カーネルの選び方を、ServiceLoader を使わない偽のプロバイダーで確かめる（CPU、native なし）。宣言が共通の収集カーネルの対応形なら、
 * デバイスのバックエンドのプロバイダーをデバイスごとに選ぶ。対応形でなければ、プロバイダーが無くても eager を選ぶ。対応形なのにプロバイダーが
 * 無いか要る演算を持たなければ、バックエンド・デバイス・プロバイダーの理由を付けた例外で止まる。選ぶ間に計画は開かない。
 */
public class CollectionPlansTest {

  /** 能力を宣言から返すか、理由だけを返す偽のプロバイダー。計画を開くと失敗する。 */
  private record Fake(Backend backend, boolean complete, String detail) implements KernelProvider {
    @Override
    public int abiVersion() {
      return ModelSchema.VERSION;
    }

    @Override
    public Capabilities capabilities(Request request) {
      return complete
          ? CollectionKernels.capabilities(request, backend)
          : new Capabilities(Set.of(), detail);
    }

    @Override
    public Plan open(Request request, long stream) {
      throw new AssertionError("Selection opened a plan");
    }
  }

  private static final List<Device> GPUS = List.of(Device.gpu(0), Device.gpu(1));

  private static Optional<CollectionPlans> select(
      InputFormat base,
      PrismShape shape,
      Precision precision,
      boolean pages,
      Backend backend,
      List<KernelProvider> providers) {
    var network = new PrismNetwork(base, shape);
    var format = network.format();
    return CollectionPlans.select(
        network.declaration(),
        GPUS,
        precision,
        KernelInputs.capacity(format, BatchPlan.Capacity.standard(format)),
        pages,
        backend,
        providers);
  }

  @Test
  public void selectsTheProviderOfTheDeviceBackend() {
    var cuda = new Fake(Backend.CUDA, true, "");
    var hip = new Fake(Backend.HIP, true, "");
    for (Backend backend : Backend.values()) {
      var expected = backend == Backend.HIP ? hip : cuda;
      for (boolean pages : List.of(false, true)) {
        var plans =
            select(
                    FOUR_PLAYER,
                    ShapeArm.V4_S.shape(),
                    Precision.BFLOAT16,
                    pages,
                    backend,
                    List.of(cuda, hip))
                .orElseThrow();
        assertEquals(plans.backend(), backend);
        assertEquals(plans.cache(), pages ? Cache.CAUSAL : Cache.OFF);
        assertEquals(plans.devices().keySet(), Set.copyOf(GPUS));
        for (Device device : GPUS) {
          var lane = plans.devices().get(device);
          assertSame(lane.provider(), expected);
          assertEquals(lane.request().device(), device.getDeviceId());
          assertEquals(lane.request().backend(), backend);
          assertEquals(lane.request().cache(), plans.cache());
        }
      }
    }
    var published =
        select(
                THREE_PLAYER,
                ShapeArm.V3.shape(),
                Precision.BFLOAT16,
                true,
                Backend.HIP,
                List.of(hip))
            .orElseThrow();
    assertEquals(published.cache(), Cache.OFF, "V3 declares no event pages");
  }

  @Test
  public void unsupportedDeclarationsRunEagerWithoutProviders() {
    for (Precision precision : Precision.values()) {
      for (PrismShape shape :
          List.of(new PrismShape.V4(512), new PrismShape.V2(384, 512, 256, 2))) {
        assertTrue(
            select(FOUR_PLAYER, shape, precision, true, Backend.HIP, List.of()).isEmpty(),
            shape + " " + precision);
      }
    }
  }

  @Test
  public void supportedDeclarationsStopWithoutAUsableProvider() {
    var shape = ShapeArm.V4_M.shape();
    var none =
        expectThrows(
            IllegalStateException.class,
            () -> select(FOUR_PLAYER, shape, Precision.BFLOAT16, false, Backend.HIP, List.of()));
    assertTrue(none.getMessage().contains("HIP"), none.getMessage());
    assertTrue(none.getMessage().contains("GPU 0"), none.getMessage());
    var other =
        expectThrows(
            IllegalStateException.class,
            () ->
                select(
                    FOUR_PLAYER,
                    shape,
                    Precision.BFLOAT16,
                    false,
                    Backend.HIP,
                    List.of(new Fake(Backend.CUDA, true, ""))));
    assertTrue(other.getMessage().contains("No PRISM HIP collection provider"), other.getMessage());
    var missing =
        expectThrows(
            IllegalStateException.class,
            () ->
                select(
                    FOUR_PLAYER,
                    shape,
                    Precision.BFLOAT16,
                    true,
                    Backend.CUDA,
                    List.of(new Fake(Backend.CUDA, false, "the fixture library is not bundled"))));
    assertTrue(
        missing.getMessage().contains("the fixture library is not bundled"), missing.getMessage());
    assertTrue(missing.getMessage().contains("EVENT_PAGES"), missing.getMessage());
  }

  @Test
  public void supportedDeclarationsRequireBfloat16() {
    var error =
        expectThrows(
            IllegalArgumentException.class,
            () ->
                select(
                    FOUR_PLAYER,
                    PrismShape.standard(),
                    Precision.FLOAT32,
                    false,
                    Backend.HIP,
                    List.of(new Fake(Backend.HIP, true, ""))));
    assertTrue(error.getMessage().contains("BFLOAT16"), error.getMessage());
  }
}
