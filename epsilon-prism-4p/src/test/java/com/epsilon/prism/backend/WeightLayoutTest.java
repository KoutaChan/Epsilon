package com.epsilon.prism.backend;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.kernel.CollectionArchitecture;
import com.epsilon.prism.kernel.ColumnIndex;
import com.epsilon.prism.kernel.EmbeddingTable;
import com.epsilon.prism.kernel.InputSection;
import com.epsilon.prism.kernel.PackedWeightLayout;
import com.epsilon.prism.kernel.TokenCounts;
import com.epsilon.prism.kernel.TokenGroup;
import com.epsilon.prism.kernel.WeightField;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.ModelShapes;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
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
  public Object[][] shapes() {
    return new Object[][] {
      {PrismShape.standard(2)},
      {PrismShape.standard(3)},
      {ShapeArm.V3.shape()},
      {ModelShapes.bidirectional(256)},
      {ShapeArm.V4_S.shape()},
      {ShapeArm.V4_M.shape()}
    };
  }

  @Test(dataProvider = "shapes")
  public void everyInferenceParameterHasASlotAndTheHeaderRoundTrips(PrismShape shape) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = HipFixtures.network(manager, shape, 11);
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, 7);
      PackView view = new PackView(pack.buffer());
      PackedWeightLayout.Header header = layout.header();
      CollectionArchitecture architecture = layout.architecture();
      Assert.assertEquals(view.architecture(), architecture.encode());
      for (ColumnIndex field : ColumnIndex.values()) {
        Assert.assertEquals(view.schema(field), header.schema(field), field.name());
      }
      for (InputSection section : InputSection.values()) {
        Assert.assertEquals(view.width(section), header.sectionWidth(section), section.name());
      }
      for (ActionRole role : FOUR_PLAYER.roles()) {
        Assert.assertEquals(
            view.composition(FOUR_PLAYER.roleIndex(role)),
            WeightLayout.composition(role),
            role.name());
        Assert.assertEquals(
            view.gate(FOUR_PLAYER.roleIndex(role)),
            WeightLayout.gateKind(FOUR_PLAYER, role),
            role.name());
      }
      for (WeightField weight : WeightField.values()) {
        for (int entry = 0; entry < weight.index().entries(); entry++) {
          if (!architecture.holds(weight, entry)) {
            Assert.assertEquals(view.weightOrAbsent(weight, entry), -1, weight + " " + entry);
            continue;
          }
          Assert.assertEquals(
              view.weight(weight, entry), header.weight(weight, entry), weight + " " + entry);
          Assert.assertEquals(view.weight(weight, entry) % PackedWeightLayout.ALIGNMENT, 0);
        }
      }
      long tables = 0;
      StringBuilder groups = new StringBuilder();
      for (EmbeddingTable.Slot slot : EmbeddingTable.Slot.values()) {
        PackView.Spec spec = view.embedding(slot);
        Assert.assertTrue(
            spec.tableOffset().length > 0
                && spec.tableOffset().length <= EmbeddingTable.MAX_TABLES);
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
          "PRISM-HIP-LAYOUT shape={} pack={} MB folded rows={} ({} MB){}",
          shape,
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
  private static void checkFold(
      PackView view, WeightLayout layout, EmbeddingTable.Slot slot, PackView.Spec spec) {
    if (slot == EmbeddingTable.Slot.TRANSITION
        || layout.architecture().hidden() != CollectionArchitecture.HIDDEN) {
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
        int at =
            spec.tableFold()[t]
                + (row * width + EmbeddingTable.foldColumn(c, attention)) * Float.BYTES;
        Assert.assertEquals(
            view.bytes.getFloat(at),
            projected(view, source, c),
            slot + " T' table " + t + " column " + c);
      }
    }
    float[] bias = new float[hidden];
    for (int h = 0; h < hidden; h++) {
      bias[h] = view.bytes.getFloat(spec.biasOffset() + h * Float.BYTES);
    }
    for (int c = 0; c < width; c++) {
      int column = EmbeddingTable.foldColumn(c, attention);
      Assert.assertEquals(
          view.bytes.getFloat(spec.fold() + (dense * width + column) * Float.BYTES),
          projected(view, bias, c),
          slot + " b'");
      Assert.assertEquals(
          view.bytes.getFloat(spec.fold() + ((dense + 1) * width + column) * Float.BYTES),
          view.f32(WeightField.QKV_BIAS, 0, c),
          slot + " b_qkv");
    }
  }

  /** (v·diag(g0)·W_qkv0ᵀ)[c] を、FP64 で h の順に足してから FP32 に丸めた値。 */
  private static float projected(PackView view, float[] v, int c) {
    double sum = 0;
    for (int h = 0; h < v.length; h++) {
      if (v[h] != 0) {
        sum +=
            v[h]
                * ((double) view.f32(WeightField.ATTENTION_NORM, 0, h)
                    * view.bf16(WeightField.QKV, 0, c * v.length + h));
      }
    }
    return (float) sum;
  }

  @Test
  public void networkConstantsMatch() {
    CollectionArchitecture architecture =
        CollectionArchitecture.from(
            new PrismNetwork(FOUR_PLAYER, PrismShape.standard()).declaration());
    Assert.assertEquals(architecture.tau(), EagerStages.tau());
    Assert.assertEquals((double) architecture.scale(), EagerStages.scale(), 0.0);
    Assert.assertEquals(
        (double) architecture.epsilon(), (float) PrismShape.standard().epsilon(), 0.0);
  }

  /**
   * 型ごとの FFN を持つ全てのアームで、局・席・牌の層のトークンの組ごとの FFN、最後の正規化の組ごとのゲイン、2 段目の 10 個のパラメーターが
   * パックの中に場所を持つことを確かめる。記述の数（四人打ちの席・牌・コード・クラス・イベント）と FFN の幅が、場所の大きさに表れることも見る。
   */
  @Test(dataProvider = "typedArms")
  public void typedArmsPlaceEveryParameter(ShapeArm arm) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismShape.V3 shape = (PrismShape.V3) arm.shape(ShapeArm.V3_DEPTH);
      PrismNetwork network = new PrismNetwork(FOUR_PLAYER, shape);
      network.initialize(manager);
      WeightLayout layout = WeightLayout.of(network);
      CollectionArchitecture architecture = layout.architecture();
      PackedWeightLayout.Header header = layout.header();
      Assert.assertEquals(architecture.counts(), new TokenCounts(4, 34, 109, 23, 110));
      Assert.assertEquals(architecture.reread(), shape.reread());
      Assert.assertEquals(architecture.fixedLayers(), 2);
      for (int layer = 1; layer < 3; layer++) {
        Assert.assertEquals(
            header.elements(WeightField.UP, WeightField.ffnEntry(layer, TokenGroup.CONTEXT)),
            (long) shape.context() * shape.hidden());
        Assert.assertEquals(
            header.elements(WeightField.UP, WeightField.ffnEntry(layer, TokenGroup.TILE)),
            (long) shape.tile() * shape.hidden());
        Assert.assertTrue(
            header.weight(WeightField.FFN_NORM, WeightField.ffnEntry(layer, TokenGroup.ALL)) < 0);
      }
      Assert.assertEquals(
          header.elements(WeightField.UP, WeightField.CANDIDATE_FFN),
          (long) shape.candidate() * shape.hidden());
      Assert.assertTrue(header.weight(WeightField.FINAL_NORM, TokenGroup.CONTEXT.ordinal()) >= 0);
      Assert.assertTrue(header.weight(WeightField.FINAL_NORM, TokenGroup.TILE.ordinal()) >= 0);
      Assert.assertEquals(header.weight(WeightField.REREAD_OUTPUT, 0) >= 0, shape.reread());
      Assert.assertEquals(
          header.elements(WeightField.REREAD_CODE_KEY, 0), 133L * shape.attention());
      WeightPack pack = WeightPack.build(layout, network, 3);
      Assert.assertEquals(new PackView(pack.buffer()).architecture(), architecture.encode());
      log.info(
          "PRISM-HIP-LAYOUT {} pack={} MB",
          arm,
          String.format(Locale.ROOT, "%.2f", layout.bytes() / 1e6));
    }
  }

  @DataProvider
  public Object[][] typedArms() {
    return new Object[][] {{ShapeArm.V3}, {ShapeArm.V3_ONE_STAGE}, {ShapeArm.V3_CONTEXT_512}};
  }
}
