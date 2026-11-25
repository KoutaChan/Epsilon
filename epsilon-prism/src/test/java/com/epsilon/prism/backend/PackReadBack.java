package com.epsilon.prism.backend;

import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.agent.observe.codec.Bfloat16;
import com.epsilon.prism.backend.WeightLayout.Conversion;
import com.epsilon.prism.backend.WeightLayout.Placement;
import com.epsilon.prism.kernel.PackedWeightLayout;
import com.epsilon.prism.kernel.WeightField;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;

/**
 * WeightPack を読み直して照合する（CPU で計算し、native は使わない）。確かめるのは次のこと。
 * 推論のパラメーターは全て、表か埋め込みのどちらかにちょうど 1 回だけ置かれている。置いた値は、変換（BF16、BF16 に丸めた FP32、FP32）の
 * とおりにビット単位で読み戻せる。形の記述にある重みは全て埋まっている。牌の関係の表は、表の入力の形式の
 * {@link com.epsilon.prism.input.TileTokens#relations()} と同じ。重み同士は重ならない。
 */
public final class PackReadBack {

  private PackReadBack() {}

  /**
   * 重みの束を読み直し、置き方、値、重みの埋まり方、牌の関係の表、重なりを照合する。
   *
   * @param layout 重みの配置の表
   * @param pack 表から作った重みの束
   * @param parameters 束を作る元にした推論の重み
   */
  public static void assertReadsBack(
      WeightLayout layout, WeightPack pack, PairList<String, Parameter> parameters) {
    ByteBuffer bytes = pack.buffer();
    Assert.assertEquals(bytes.remaining(), layout.bytes());
    PackedWeightLayout.Header header = layout.header();
    Map<String, float[]> values = new HashMap<>();
    for (Pair<String, Parameter> pair : parameters) {
      values.put(pair.getKey(), pair.getValue().getArray().toFloatArray());
    }
    Set<String> placed = new HashSet<>();
    Set<String> entries = new HashSet<>();
    List<long[]> extents = new ArrayList<>();
    for (Placement placement : layout.placements()) {
      String name = placement.parameter();
      float[] source = values.get(name);
      Assert.assertNotNull(source, name);
      Assert.assertTrue(placed.add(name), name + " is placed twice");
      if (placement.weight() != null) {
        Assert.assertTrue(
            entries.add(placement.weight() + "/" + placement.entry()),
            placement + " shares an entry");
        Assert.assertEquals(
            header.weight(placement.weight(), placement.entry()), placement.offset(), name);
        Assert.assertEquals(
            header.elements(placement.weight(), placement.entry()), source.length, name);
        extents.add(
            new long[] {
              placement.offset(),
              placement.offset() + header.bytes(placement.weight(), placement.entry())
            });
      }
      int at = Math.toIntExact(placement.offset());
      for (int i = 0; i < source.length; i++) {
        switch (placement.conversion()) {
          case BF16 -> {
            if (bytes.getShort(at + i * Short.BYTES) != Bfloat16.fromFloat(source[i])) {
              Assert.fail(name + "[" + i + "] does not read back as BF16");
            }
          }
          case ROUNDED -> {
            if (Float.floatToRawIntBits(bytes.getFloat(at + i * Float.BYTES))
                != Float.floatToRawIntBits(WeightPack.round(source[i]))) {
              Assert.fail(name + "[" + i + "] does not read back rounded to BF16");
            }
          }
          case FP32 -> {
            if (Float.floatToRawIntBits(bytes.getFloat(at + i * Float.BYTES))
                != Float.floatToRawIntBits(source[i])) {
              Assert.fail(name + "[" + i + "] does not read back as FP32");
            }
          }
        }
      }
    }
    assertCandidateTables(layout, values, extents);
    for (WeightLayout.Embedding embedding : layout.embeddings().values()) {
      for (String name : List.of(embedding.table(), embedding.projection(), embedding.bias())) {
        Assert.assertTrue(values.containsKey(name), name);
        Assert.assertTrue(placed.add(name), name + " is placed twice");
      }
    }
    Assert.assertEquals(placed, values.keySet(), "every inference parameter is placed once");

    for (WeightField weight : WeightField.values()) {
      for (int entry = 0; entry < weight.index().entries(); entry++) {
        boolean held = layout.architecture().holds(weight, entry);
        Assert.assertEquals(header.weight(weight, entry) >= 0, held, weight + " " + entry);
        if (held && weight != WeightField.TILE_RELATION) {
          Assert.assertTrue(
              entries.contains(weight + "/" + entry), weight + " " + entry + " has no parameter");
        }
      }
    }
    long tiles = header.weight(WeightField.TILE_RELATION, 0);
    int[] relations = layout.format().tileTokens().relations();
    Assert.assertEquals(header.elements(WeightField.TILE_RELATION, 0), relations.length);
    for (int i = 0; i < relations.length; i++) {
      Assert.assertEquals(
          bytes.get(Math.toIntExact(tiles) + i) & 0xFF, relations[i], "tile relation " + i);
    }
    extents.add(new long[] {tiles, tiles + relations.length});

    extents.sort(Comparator.comparingLong(extent -> extent[0]));
    long end = PackedWeightLayout.HEADER_BYTES;
    for (long[] extent : extents) {
      Assert.assertTrue(
          extent[0] >= end, "weight at " + extent[0] + " overlaps the one ending at " + end);
      Assert.assertEquals(extent[0] % PackedWeightLayout.ALIGNMENT, 0);
      end = extent[1];
    }
    Assert.assertTrue(end <= layout.bytes());
  }

  /** 末尾の候補表は、共通ヘッダーの欄ではなく、収集計画が持つ位置から読む。 */
  private static void assertCandidateTables(
      WeightLayout layout, Map<String, float[]> values, List<long[]> extents) {
    List<Placement> appended =
        layout.placements().stream().filter(p -> p.weight() == null).toList();
    var options = layout.planOptions();
    var count = options.count();
    if (count == null) {
      Assert.assertNull(options.segments());
      Assert.assertTrue(appended.isEmpty(), "unexpected candidate tables");
      return;
    }
    var segments = options.segments();
    Assert.assertNotNull(segments, "missing candidate segment table");
    var segmentLayout = segments.layout();
    Assert.assertEquals(appended.size(), 3 + segmentLayout.spans().size());

    long first = count.offset();
    long gateElements = (long) (layout.format().keys().codes() - count.firstCode()) * count.width();
    long tileElements = (long) layout.architecture().hidden() * count.width();
    long end = assertPlacement(appended.get(0), values, first, gateElements, Conversion.FP32);
    end = assertPlacement(appended.get(1), values, end, tileElements, Conversion.BF16);
    end = assertPlacement(appended.get(2), values, end, tileElements, Conversion.BF16);
    extents.add(new long[] {first, end});

    first = segments.offset();
    long rowElements = 2L * layout.architecture().attention();
    long rowBytes = rowElements * Float.BYTES;
    for (int i = 0; i < segmentLayout.spans().size(); i++) {
      var span = segmentLayout.spans().get(i);
      assertPlacement(
          appended.get(3 + i),
          values,
          first + span.firstRow() * rowBytes,
          span.rows() * rowElements,
          Conversion.FP32);
    }
    extents.add(new long[] {first, first + segmentLayout.tableRows() * rowBytes});
  }

  private static long assertPlacement(
      Placement placement,
      Map<String, float[]> values,
      long offset,
      long elements,
      Conversion conversion) {
    String name = placement.parameter();
    Assert.assertEquals(placement.offset(), offset, name);
    Assert.assertEquals(placement.conversion(), conversion, name);
    Assert.assertEquals((long) values.get(name).length, elements, name);
    return offset + elements * (conversion == Conversion.BF16 ? Short.BYTES : Float.BYTES);
  }
}
