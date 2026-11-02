package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipPack;
import com.epsilon.prism.hip.HipSchema;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * WeightLayout と WeightPack の形（CPU、native なし）: 全ての推論のパラメーターが置き場を持つ、頭を native の並びで読み直すと同じ値、
 * 畳んだ組の表が 8 引き以下、網の定数（τ・尺度）と同じ値。
 */
public class WeightLayoutTest {
  private static final Logger log = LoggerFactory.getLogger(WeightLayoutTest.class);

  @DataProvider
  public Object[][] depths() {
    return new Object[][] {{2}, {3}};
  }

  @Test(dataProvider = "depths")
  public void everyInferenceParameterHasASlotAndTheHeaderRoundTrips(int depth) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = HipFixtures.network(manager, depth, 11);
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, 7);
      PackView view = new PackView(pack.buffer());
      HipPack.Header header = layout.header();
      for (HipSchema field : HipSchema.values()) {
        Assert.assertEquals(view.schema(field), header.schema(field), field.name());
      }
      for (HipSection section : HipSection.values()) {
        Assert.assertEquals(view.width(section), header.sectionWidth(section), section.name());
      }
      for (ActionRole role : ActionRole.values()) {
        Assert.assertEquals(view.composition(role.ordinal()), WeightLayout.composition(role), role.name());
        Assert.assertEquals(view.gate(role.ordinal()), WeightLayout.gateKind(role), role.name());
      }
      for (HipWeight weight : HipWeight.values()) {
        for (int layer = 0; layer < (weight.layered() ? depth : 1); layer++) {
          Assert.assertEquals(view.weight(weight, layer), header.weight(weight, layer), weight + " " + layer);
          Assert.assertEquals(view.weight(weight, layer) % HipPack.ALIGNMENT, 0);
        }
      }
      long tables = 0;
      StringBuilder groups = new StringBuilder();
      for (HipEmbedding.Slot slot : HipEmbedding.Slot.values()) {
        PackView.Spec spec = view.embedding(slot);
        Assert.assertTrue(spec.tableOffset().length > 0 && spec.tableOffset().length <= HipEmbedding.MAX_TABLES);
        groups.append(' ').append(slot).append('=');
        for (int rows : spec.tableRows()) {
          tables += rows;
          groups.append(rows).append('/');
        }
        Assert.assertEquals(spec.source().length, header.embedding(slot).dense().size());
        checkFold(view, layout, slot, spec);
      }
      Assert.assertEquals(pack.version(), 7);
      Assert.assertEquals(pack.buffer().remaining(), layout.bytes());
      log.info(
          "PRISM-HIP-LAYOUT depth={} pack={} MB folded rows={} ({} MB){}",
          depth,
          String.format(Locale.ROOT, "%.2f", layout.bytes() / 1e6),
          tables,
          String.format(Locale.ROOT, "%.2f", tables * PrismShape.standard().hidden() * 4 / 1e6),
          groups);
    }
  }

  /**
   * 層 0 の QKV に畳んだ表（L0-FOLD）: 遷移の型にはなく、ほかの型は T' の各表の行 1 つと b'・b_qkv を FP64 で作り直した値と同じ（列は頭の
   * 対の並び）。
   */
  private static void checkFold(PackView view, WeightLayout layout, HipEmbedding.Slot slot, PackView.Spec spec) {
    if (slot == HipEmbedding.Slot.TRANSITION) {
      Assert.assertEquals(spec.fold(), -1);
      for (int offset : spec.tableFold()) {
        Assert.assertEquals(offset, -1);
      }
      return;
    }
    int hidden = layout.shape().hidden();
    int attention = layout.shape().attention();
    int width = 3 * attention;
    int dense = spec.source().length;
    for (int t = 0; t < spec.tableOffset().length; t++) {
      int row = spec.tableRows()[t] / 2;
      float[] source = new float[hidden];
      for (int h = 0; h < hidden; h++) {
        source[h] = view.bytes.getFloat(spec.tableOffset()[t] + (row * hidden + h) * Float.BYTES);
      }
      for (int c = 0; c < width; c++) {
        int at = spec.tableFold()[t] + (row * width + HipEmbedding.foldColumn(c, attention)) * Float.BYTES;
        Assert.assertEquals(view.bytes.getFloat(at), projected(view, source, c), slot + " T' table " + t + " column " + c);
      }
    }
    float[] bias = new float[hidden];
    for (int h = 0; h < hidden; h++) {
      bias[h] = view.bytes.getFloat(spec.biasOffset() + h * Float.BYTES);
    }
    for (int c = 0; c < width; c++) {
      int column = HipEmbedding.foldColumn(c, attention);
      Assert.assertEquals(
          view.bytes.getFloat(spec.fold() + (dense * width + column) * Float.BYTES), projected(view, bias, c), slot + " b'");
      Assert.assertEquals(
          view.bytes.getFloat(spec.fold() + ((dense + 1) * width + column) * Float.BYTES),
          view.f32(HipWeight.QKV_BIAS, 0, c),
          slot + " b_qkv");
    }
  }

  /** (v·diag(g0)·W_qkv0ᵀ)[c] を FP64 で h の順に足して FP32 に丸めた値。 */
  private static float projected(PackView view, float[] v, int c) {
    double sum = 0;
    for (int h = 0; h < v.length; h++) {
      if (v[h] != 0) {
        sum += v[h] * ((double) view.f32(HipWeight.ATTENTION_NORM, 0, h) * view.bf16(HipWeight.QKV, 0, c * v.length + h));
      }
    }
    return (float) sum;
  }

  @Test
  public void networkConstantsMatch() {
    Assert.assertEquals(WeightLayout.TAU, EagerStages.tau());
    Assert.assertEquals((double) WeightLayout.SCALE, EagerStages.scale(), 0.0);
  }
}
