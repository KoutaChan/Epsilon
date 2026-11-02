package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.model.PrismNetwork;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 参照の既定の層 0（K1 の L0-FOLD）が式 D と合う（CPU）。D は試験の側で束の T・W_d・b と層 0 の g・W_qkv から作り直す:
 * qkv0 = ⌊r·(ΣT'[idx] + ΣW'_d·f + b') + b_qkv⌉、T' = T·diag(g)·W_qkvᵀ・W'_d・b' は FP64 で作って FP32 に丸めた値、r は丸める前の x の
 * 1/rms（k の順の FP32 の和）。参照（束の畳んだ表とカーネルと同じ FP32 の順）とは r の和の順だけが違うので、全ての値が BF16 の 1 ulp
 * 以内で、ビットが違う値の割合が {@value #DIFFER_LIMIT} 以下であることを求める。網は乱数で動かした深さ 2 の網、行は価値グラフの全ての作り方が
 * 出る行。
 */
public class HipFoldReferenceTest {
  private static final Logger log = LoggerFactory.getLogger(HipFoldReferenceTest.class);

  private static final int ROWS = 64;
  private static final int DEPTH = 2;
  private static final long SEED = 4300;
  private static final double DIFFER_LIMIT = 1e-3;

  @Test
  public void defaultLayer0IsFormD() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      PrismNetwork network = HipFixtures.network(manager, DEPTH, SEED);
      WeightPack pack = WeightPack.build(WeightLayout.of(network), network, 1);
      HipReference reference = new HipReference(pack);
      HipReference.Input in = new HipReference.Input(HipFixtures.encode(HipFixtures.mixedRows(ROWS, SEED)));
      reference.allocate(in);
      reference.embed(in);
      float[] expected = formD(reference, in);
      float[] actual = reference.layer0Qkv();
      int differ = 0;
      for (int i = 0; i < expected.length; i++) {
        differ += Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(actual[i]) ? 1 : 0;
      }
      Diff diff = Diff.bf16(expected, actual);
      log.info("PRISM-HIP-FOLD {}, bits differ {}", diff.line("reference qkv0 vs D"), differ);
      Assert.assertTrue(diff.withinOneUlp() == 1.0, "qkv0 values beyond 1 BF16 ulp of D: " + diff.line("qkv0"));
      Assert.assertTrue(
          differ <= DIFFER_LIMIT * expected.length, differ + " of " + expected.length + " qkv0 values differ from D");
    }
  }

  /** 式 D の qkv0（[トークン, 3A]、列は頭の対の並び。参照の {@link HipReference#layer0Qkv} と同じ並び）。 */
  private static float[] formD(HipReference r, HipReference.Input in) {
    int hidden = r.hidden;
    int attention = r.attention;
    int width = 3 * attention;
    double[] projection = new double[hidden * width];
    for (int h = 0; h < hidden; h++) {
      double gain = r.weight(HipWeight.ATTENTION_NORM, 0, h);
      for (int c = 0; c < width; c++) {
        projection[h * width + c] = gain * r.weight(HipWeight.QKV, 0, c * hidden + h);
      }
    }
    PackView pack = r.pack;
    float[] out = new float[(r.fixed * in.rows + in.events) * width];
    for (HipEmbedding.Slot slot : HipEmbedding.Slot.values()) {
      if (slot == HipEmbedding.Slot.TRANSITION) {
        continue;
      }
      PackView.Spec spec = pack.embedding(slot);
      int tables = spec.tableOffset().length;
      int features = spec.source().length;
      float[][] dense = new float[features][];
      double[] source = new double[hidden];
      for (int k = 0; k < features; k++) {
        for (int h = 0; h < hidden; h++) {
          source[h] = bf16(pack, spec.denseOffset() + (k * hidden + h) * Short.BYTES);
        }
        dense[k] = project(source, projection, width);
      }
      for (int h = 0; h < hidden; h++) {
        source[h] = pack.bytes.getFloat(spec.biasOffset() + h * Float.BYTES);
      }
      float[] bias = project(source, projection, width);
      List<Map<Integer, float[]>> folded = new ArrayList<>();
      for (int table = 0; table < tables; table++) {
        folded.add(new HashMap<>());
      }
      Tensor category = HipReference.tensor(spec.category());
      Tensor scalar = spec.scalar() < 0 ? null : HipReference.tensor(spec.scalar());
      Tensor exact = spec.exact() < 0 ? null : HipReference.tensor(spec.exact());
      int count = r.slotCount(slot, in);
      int start = r.slotStart(slot, in);
      for (int t = 0; t < count; t++) {
        float[] acc = new float[hidden];
        float[] sum = new float[width];
        for (int table = 0; table < tables; table++) {
          int entry = 0;
          for (int m = 0; m < spec.columns()[table].length; m++) {
            entry += in.u8(category, t, spec.columns()[table][m]) * spec.strides()[table][m];
          }
          int base = spec.tableOffset()[table] + entry * hidden * Float.BYTES;
          for (int h = 0; h < hidden; h++) {
            acc[h] += pack.bytes.getFloat(base + h * Float.BYTES);
          }
          float[] row =
              folded.get(table).computeIfAbsent(entry, e -> {
                double[] values = new double[hidden];
                for (int h = 0; h < hidden; h++) {
                  values[h] = pack.bytes.getFloat(base + h * Float.BYTES);
                }
                return project(values, projection, width);
              });
          for (int c = 0; c < width; c++) {
            sum[c] += row[c];
          }
        }
        for (int k = 0; k < features; k++) {
          int index = spec.source()[k] & 0xFFFF;
          float feature =
              switch (HipEmbedding.Source.values()[spec.source()[k] >> 16]) {
                case FLAG -> in.u8(category, t, index);
                case SCALAR -> in.bf16(scalar, t, index);
                case EXACT -> HipReference.bf(in.f32(exact, t, index));
              };
          for (int h = 0; h < hidden; h++) {
            acc[h] = Math.fma(bf16(pack, spec.denseOffset() + (k * hidden + h) * Short.BYTES), feature, acc[h]);
          }
          for (int c = 0; c < width; c++) {
            sum[c] = Math.fma(dense[k][c], feature, sum[c]);
          }
        }
        float squares = 0f;
        for (int h = 0; h < hidden; h++) {
          acc[h] += pack.bytes.getFloat(spec.biasOffset() + h * Float.BYTES);
          squares += acc[h] * acc[h];
        }
        float rms = (float) (1.0 / Math.sqrt(squares / hidden + r.epsilon));
        for (int c = 0; c < width; c++) {
          out[(start + t) * width + HipEmbedding.foldColumn(c, attention)] =
              HipReference.bf(rms * (sum[c] + bias[c]) + r.weight(HipWeight.QKV_BIAS, 0, c));
        }
      }
    }
    return out;
  }

  /** v·M を FP64 で h の順に足して FP32 に丸めた値（M は {@code [v の長さ, width]}）。 */
  private static float[] project(double[] v, double[] m, int width) {
    double[] sum = new double[width];
    for (int h = 0; h < v.length; h++) {
      for (int c = 0; c < width; c++) {
        sum[c] += v[h] * m[h * width + c];
      }
    }
    float[] out = new float[width];
    for (int c = 0; c < width; c++) {
      out[c] = (float) sum[c];
    }
    return out;
  }

  private static float bf16(PackView pack, int offset) {
    return Float.intBitsToFloat((pack.bytes.getShort(offset) & 0xFFFF) << 16);
  }
}
