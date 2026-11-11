package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipCounts;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipPack;
import com.epsilon.prism.hip.HipSchema;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.hip.TokenGroup;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism4p.input.InputSchema;
import com.epsilon.prism4p.input.InputSchema.ActionRole;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * WeightLayout と WeightPack の形を確かめる（CPU、native なし）。推論に使う全てのパラメーターがパックの中に場所を持つこと、ヘッダーを
 * native の並びで読み直すと同じ値になること、埋め込みの型ごとに組の表（W·E を折り込んだ表）が 8 つ以下であること、ネットワークの定数
 * （τ・スケール）が eager と同じ値であることを見る。
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
      HipArchitecture architecture = layout.architecture();
      Assert.assertEquals(view.architecture(), architecture.encode());
      Assert.assertEquals(architecture.depth(), depth);
      for (HipSchema field : HipSchema.values()) {
        Assert.assertEquals(view.schema(field), header.schema(field), field.name());
      }
      for (HipSection section : HipSection.values()) {
        Assert.assertEquals(view.width(section), header.sectionWidth(section), section.name());
      }
      for (ActionRole role : ActionRole.values()) {
        Assert.assertEquals(view.composition(role.ordinal()), WeightLayout.composition(role), role.name());
        Assert.assertEquals(view.gate(role.ordinal()), WeightLayout.gateKind(InputSchema.FORMAT, role), role.name());
      }
      for (HipWeight weight : HipWeight.values()) {
        for (int entry = 0; entry < weight.index().entries(); entry++) {
          if (!architecture.holds(weight, entry)) {
            Assert.assertEquals(view.weightOrAbsent(weight, entry), -1, weight + " " + entry);
            continue;
          }
          Assert.assertEquals(view.weight(weight, entry), header.weight(weight, entry), weight + " " + entry);
          Assert.assertEquals(view.weight(weight, entry) % HipPack.ALIGNMENT, 0);
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
   * 層 0 の QKV に折り込んだ表（L0-FOLD）を確かめる。遷移の型には無い。ほかの型では、T' の各表の行 1 つと b'・b_qkv が、FP64 で
   * 作り直した値と一致する（列は Attention のヘッドの対の並び）。
   */
  private static void checkFold(PackView view, WeightLayout layout, HipEmbedding.Slot slot, PackView.Spec spec) {
    if (slot == HipEmbedding.Slot.TRANSITION) {
      Assert.assertEquals(spec.fold(), -1);
      for (int offset : spec.tableFold()) {
        Assert.assertEquals(offset, -1);
      }
      return;
    }
    int hidden = layout.architecture().hidden();
    int attention = layout.architecture().attention();
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

  /** (v·diag(g0)·W_qkv0ᵀ)[c] を、FP64 で h の順に足してから FP32 に丸めた値。 */
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
    HipArchitecture architecture = PrismShape.standard().architecture(InputSchema.FORMAT);
    Assert.assertEquals(architecture.tau(), EagerStages.tau());
    Assert.assertEquals((double) architecture.scale(), EagerStages.scale(), 0.0);
    Assert.assertEquals((double) architecture.epsilon(), (float) PrismShape.standard().epsilon(), 0.0);
  }

  /**
   * 型ごとの FFN を持つ全てのアームで、局・席・牌の層のトークンの組ごとの FFN、最後の正規化の組ごとのゲイン、2 段目の 10 個のパラメーターが
   * パックの中に場所を持つことを確かめる。記述の数（四人打ちの席・牌・コード・クラス・イベント）と FFN の幅が、場所の大きさに表れることも見る。
   */
  @Test(dataProvider = "typedArms")
  public void typedArmsPlaceEveryParameter(ShapeArm arm) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismShape shape = arm.shape(ShapeArm.V3_DEPTH);
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
      network.initialize(manager);
      WeightLayout layout = WeightLayout.of(network);
      HipArchitecture architecture = layout.architecture();
      HipPack.Header header = layout.header();
      Assert.assertEquals(architecture.counts(), new HipCounts(4, 34, 109, 23, 110));
      Assert.assertEquals(architecture.reread(), shape.reread());
      Assert.assertEquals(architecture.fixedLayers(), 2);
      for (int layer = 1; layer < 3; layer++) {
        Assert.assertEquals(
            header.elements(HipWeight.UP, HipWeight.ffnEntry(layer, TokenGroup.CONTEXT)),
            (long) shape.typed().context() * shape.hidden());
        Assert.assertEquals(
            header.elements(HipWeight.UP, HipWeight.ffnEntry(layer, TokenGroup.TILE)),
            (long) shape.typed().tile() * shape.hidden());
        Assert.assertTrue(header.weight(HipWeight.FFN_NORM, HipWeight.ffnEntry(layer, TokenGroup.ALL)) < 0);
      }
      Assert.assertEquals(
          header.elements(HipWeight.UP, HipWeight.CANDIDATE_FFN), (long) shape.typed().candidate() * shape.hidden());
      Assert.assertTrue(header.weight(HipWeight.FINAL_NORM, TokenGroup.CONTEXT.ordinal()) >= 0);
      Assert.assertTrue(header.weight(HipWeight.FINAL_NORM, TokenGroup.TILE.ordinal()) >= 0);
      Assert.assertEquals(header.weight(HipWeight.REREAD_OUTPUT, 0) >= 0, shape.reread());
      Assert.assertEquals(header.elements(HipWeight.REREAD_CODE_KEY, 0), 133L * shape.attention());
      WeightPack pack = WeightPack.build(layout, network, 3);
      Assert.assertEquals(new PackView(pack.buffer()).architecture(), architecture.encode());
      log.info("PRISM-HIP-LAYOUT {} pack={} MB", arm, String.format(Locale.ROOT, "%.2f", layout.bytes() / 1e6));
    }
  }

  @DataProvider
  public Object[][] typedArms() {
    return new Object[][] {{ShapeArm.V3}, {ShapeArm.V3_ONE_STAGE}, {ShapeArm.V3_CONTEXT_512}};
  }
}
