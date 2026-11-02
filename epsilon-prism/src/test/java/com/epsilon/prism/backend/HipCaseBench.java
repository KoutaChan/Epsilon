package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.HipShape;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.model.PrismNetwork;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * JVM の外の台（epsilon-prism-hip の prism_hip_bench）に渡す 1 塊の材料を書く。網・行・容量・版は {@link HipForwardTest#measureSpeed} の
 * 深さ 2 と同じなので、台のカーネルの時間はその記録の行とそのまま比べられる。
 *
 * <p>書く先は {@code -Dprism.hip.case=<dir>}（無ければ飛ばす）。中身は case.txt（計画の形・塊の数・区画の位置・版、"名前=値"）、
 * weights.bin（WeightPack の頭つきの全体）、input.bin（blob の先頭 {@link HipBatch#bytes()} バイト）。{@link #writeScores} は加えて
 * scores.bin（{@link HipBackend#forward} の s、要求の順の FP32 little-endian）を書く。どれも種が決まっているので、何度書いても
 * バイトで同じ。名前が *Test でないので、既定の試験には入らず {@code -Dtest} で名指ししたときだけ走る。
 */
public class HipCaseBench {
  private static final Logger log = LoggerFactory.getLogger(HipCaseBench.class);

  private static final String DIRECTORY_PROPERTY = "prism.hip.case";
  private static final int DEPTH = 2;
  private static final int ROWS = 2048;
  private static final long ROW_SEED = 800;
  private static final long NETWORK_SEED = 70 + DEPTH;
  private static final long VERSION = 5;

  /** 材料だけを書く（CPU だけ。GPU は開かない）。 */
  @Test
  public void writeCase() {
    Path directory = directory();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Case material = Case.build(manager);
      material.write(directory);
      log.info("PRISM-HIP-CASE wrote {} ({})", directory, material.batch());
    }
  }

  /** 材料と、同じ塊の {@link HipBackend#forward} の s を書く（GPU 1 枚）。 */
  @Test(groups = "rocm")
  public void writeScores() {
    Path directory = directory();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0))) {
      Case material = Case.build(manager);
      material.write(directory);
      float[] scores;
      try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, material.layout(), material.capacity())) {
        backend.publish(material.pack());
        scores = backend.forward(material.host(), VERSION);
      }
      ByteBuffer bytes =
          ByteBuffer.allocate(Math.multiplyExact(scores.length, Float.BYTES)).order(ByteOrder.LITTLE_ENDIAN);
      bytes.asFloatBuffer().put(scores);
      write(directory.resolve("scores.bin"), bytes);
      log.info("PRISM-HIP-CASE wrote {} scores to {}", scores.length, directory);
    }
  }

  private static Path directory() {
    String given = System.getProperty(DIRECTORY_PROPERTY);
    if (given == null || given.isBlank()) {
      throw new SkipException("set -D" + DIRECTORY_PROPERTY + "=<dir> to write the bench case");
    }
    return Path.of(given);
  }

  private static void write(Path file, ByteBuffer content) {
    try (FileChannel channel =
        FileChannel.open(
            file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
      while (content.hasRemaining()) {
        channel.write(content);
      }
    } catch (IOException e) {
      throw new UncheckedIOException("cannot write " + file, e);
    }
  }

  /**
   * 1 塊の材料。
   *
   * @param layout 網の表
   * @param pack 重みの束
   * @param capacity 計画の容量（measureSpeed と同じ）
   * @param shape 計画の形
   * @param host 封じた塊
   * @param batch 塊の数と区画の位置
   */
  private record Case(
      WeightLayout layout,
      WeightPack pack,
      BatchPlan.Capacity capacity,
      HipShape shape,
      HostBatch host,
      HipBatch batch) {

    static Case build(NDManager manager) {
      PrismNetwork network = HipFixtures.network(manager, DEPTH, NETWORK_SEED);
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, VERSION);
      BatchPlan.Capacity capacity = GpuCase.capacity(ROWS);
      HostBatch host = HipFixtures.encode(HipFixtures.simulatedRows(ROWS, ROW_SEED));
      return new Case(layout, pack, capacity, layout.hipShape(capacity), host, HipBackend.batchOf(host));
    }

    void write(Path directory) {
      try {
        Files.createDirectories(directory);
        Files.write(directory.resolve("case.txt"), description(), StandardCharsets.UTF_8);
      } catch (IOException e) {
        throw new UncheckedIOException("cannot write the case description to " + directory, e);
      }
      HipCaseBench.write(directory.resolve("weights.bin"), pack.buffer());
      ByteBuffer input = host.blob().duplicate().position(0).limit(Math.toIntExact(batch.bytes()));
      HipCaseBench.write(directory.resolve("input.bin"), input);
    }

    /** case.txt の行（prism_hip_bench が読む名前。ε はビットで渡す）。 */
    private List<String> description() {
      HipShape.Capacity planned = shape.capacity();
      List<String> lines = new ArrayList<>();
      lines.add("hidden=" + shape.hidden());
      lines.add("ffn=" + shape.ffn());
      lines.add("gate=" + shape.gate());
      lines.add("depth=" + shape.depth());
      lines.add("epsilonBits=" + Integer.toUnsignedString(Float.floatToRawIntBits(shape.epsilon())));
      lines.add("fixedTokens=" + shape.fixedTokens());
      lines.add("capacityRows=" + planned.rows());
      lines.add("capacityEvents=" + planned.events());
      lines.add("capacityTransitions=" + planned.transitions());
      lines.add("capacityActions=" + planned.actions());
      lines.add("capacityInputBytes=" + planned.inputBytes());
      lines.add("weightBytes=" + shape.weightBytes());
      lines.add("version=" + VERSION);
      lines.add("rows=" + batch.rows());
      lines.add("events=" + batch.events());
      lines.add("transitions=" + batch.transitions());
      lines.add("actions=" + batch.actions());
      lines.add("bytes=" + batch.bytes());
      StringBuilder sections = new StringBuilder();
      for (long offset : batch.sections()) {
        sections.append(sections.isEmpty() ? "" : ",").append(offset);
      }
      lines.add("sections=" + sections);
      return lines;
    }
  }
}
