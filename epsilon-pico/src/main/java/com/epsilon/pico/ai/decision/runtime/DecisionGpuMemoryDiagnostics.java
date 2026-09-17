package com.epsilon.pico.ai.decision.runtime;

import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtAllocatorSnapshot;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtMemoryStats;
import com.epsilon.config.settings.DecisionPerfSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionDeviceStreams;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 完了した処理のメモリ割り当て情報を取得し、構造化ログへ出力する。 */
public final class DecisionGpuMemoryDiagnostics {

  private static final Logger log = LoggerFactory.getLogger(DecisionGpuMemoryDiagnostics.class);

  private DecisionGpuMemoryDiagnostics() {}

  public static void logSnapshot(String phase, DecisionExecutionContext context) {
    logSnapshot(phase, context, EpsilonSettings.defaults().bind(DecisionPerfSettings.class));
  }

  public static void logSnapshot(
      String phase, DecisionExecutionContext context, DecisionPerfSettings settings) {
    if (!settings.memorySnapshots()) {
      return;
    }
    context.visitStreams(streams -> logDevice(phase, streams));
  }

  private static void logDevice(String phase, DecisionDeviceStreams streams) {
    PtEngine engine = (PtEngine) Engine.getEngine("PyTorch");
    PtMemoryStats memory = engine.getMemoryStats(streams.device());
    PtAllocatorSnapshot snapshot = engine.getAllocatorSnapshot(streams.device());
    JsonObject report = new JsonObject();
    report.addProperty("phase", phase);
    report.addProperty("device", streams.device().getDeviceId());
    JsonObject roles = new JsonObject();
    roles.addProperty("compute", token(streams.compute().getId()));
    roles.addProperty("h2d", token(streams.h2d().getId()));
    roles.addProperty("d2h", token(streams.d2h().getId()));
    report.add("streams", roles);
    report.addProperty("allocatedBytes", memory.getAllocatedBytes());
    report.addProperty("reservedBytes", memory.getReservedBytes());
    report.addProperty("activeBytes", memory.getActiveBytes());
    report.addProperty("peakAllocatedBytes", memory.getPeakAllocatedBytes());
    report.addProperty("peakReservedBytes", memory.getPeakReservedBytes());
    report.addProperty("peakActiveBytes", memory.getPeakActiveBytes());
    report.addProperty("inactiveSplitBytes", memory.getInactiveSplitBytes());
    report.addProperty("numAllocRetries", memory.getAllocationRetries());
    report.addProperty("numOoms", memory.getOutOfMemoryCount());
    JsonArray pools = new JsonArray();
    for (PtAllocatorSnapshot.StreamPool pool : snapshot.getPools()) {
      JsonObject row = new JsonObject();
      row.addProperty("streamToken", token(pool.getStreamId()));
      row.addProperty("poolIdHigh", Long.toUnsignedString(pool.getPoolIdHigh()));
      row.addProperty("poolIdLow", Long.toUnsignedString(pool.getPoolIdLow()));
      row.addProperty("large", pool.isLarge());
      row.addProperty("reservedBytes", pool.getReservedBytes());
      row.addProperty("allocatedBytes", pool.getAllocatedBytes());
      row.addProperty("activeBytes", pool.getActiveBytes());
      row.addProperty("largestInactiveBlockBytes", pool.getLargestInactiveBlockBytes());
      row.addProperty("segmentCount", pool.getSegmentCount());
      pools.add(row);
    }
    report.add("pools", pools);
    log.info("Decision GPU memory: {}", report);
  }

  private static String token(long value) {
    return "0x" + Long.toUnsignedString(value, 16);
  }
}
