package com.epsilon.runtime;

import jdk.jfr.Category;
import jdk.jfr.DataAmount;
import jdk.jfr.Enabled;
import jdk.jfr.Event;
import jdk.jfr.EventType;
import jdk.jfr.Label;
import jdk.jfr.Name;
import jdk.jfr.StackTrace;
import jdk.jfr.Timespan;

/** 入力の選択、符号化、転送準備をバッチ単位で記録する。明示的に有効化したJFR収録だけで使用する。 */
public final class InputBatchProfile {
  private static final EventType SELECTION = EventType.getEventType(Selection.class);
  private static final EventType ENCODING = EventType.getEventType(Encoding.class);
  private static final EventType STAGE = EventType.getEventType(Stage.class);
  private static final EventType COLLECTION = EventType.getEventType(Collection.class);

  private InputBatchProfile() {}

  public static Selection beginSelection() {
    if (!SELECTION.isEnabled()) return null;
    Selection event = new Selection();
    event.begin();
    return event;
  }

  public static Encoding beginEncoding(
      Object bucket, int rows, int taskCount, int taskIndex, String phase) {
    if (!ENCODING.isEnabled()) return null;
    Encoding event = new Encoding();
    event.bucket = String.valueOf(bucket);
    event.rows = rows;
    event.batchRows = rows;
    event.taskCount = taskCount;
    event.taskIndex = taskIndex;
    event.phase = phase;
    event.begin();
    return event;
  }

  public static Stage beginStage(String series, Object bucket, int rows, String phase) {
    if (!STAGE.isEnabled()) return null;
    Stage event = new Stage();
    event.series = series;
    event.bucket = String.valueOf(bucket);
    event.rows = rows;
    event.phase = phase;
    event.begin();
    return event;
  }

  public static Collection beginCollection(int games, int warmupGames) {
    if (!COLLECTION.isEnabled()) return null;
    Collection event = new Collection();
    event.games = games;
    event.warmupGames = warmupGames;
    event.begin();
    return event;
  }

  @Name("epsilon.InputSelection")
  @Label("Input batch selection")
  @Category("Epsilon")
  @Enabled(false)
  @StackTrace(false)
  public static final class Selection extends Event {
    public boolean admitted;
    public String model;
    public String bucket;
    public String dispatchReason;
    public int rows;
    public int capacity;

    @Timespan(Timespan.NANOSECONDS)
    public long oldestRowWaitNanos;

    @Timespan(Timespan.NANOSECONDS)
    public long totalRowWaitNanos;
  }

  @Name("epsilon.InputEncoding")
  @Label("Input batch encoding")
  @Category("Epsilon")
  @Enabled(false)
  @StackTrace(false)
  public static final class Encoding extends Event {
    public String bucket;
    public String phase;
    public int rows;
    public int batchRows;
    public int taskCount;
    public int taskIndex;
    public boolean success;

    @Timespan(Timespan.NANOSECONDS)
    public long queueWaitNanos;

    @Timespan(Timespan.NANOSECONDS)
    public long runNanos;

    @Timespan(Timespan.NANOSECONDS)
    public long threadCpuNanos = -1;
  }

  @Name("epsilon.InputStage")
  @Label("Input batch staging")
  @Category("Epsilon")
  @Enabled(false)
  @StackTrace(false)
  public static final class Stage extends Event implements AutoCloseable {
    public String series;
    public String bucket;
    public String phase;
    public int rows;
    public boolean success;

    @DataAmount(DataAmount.BYTES)
    public long categoricalBytes;

    @DataAmount(DataAmount.BYTES)
    public long numericBytes;

    @DataAmount(DataAmount.BYTES)
    public long indexBytes;

    @Override
    public void close() {
      end();
      commit();
    }
  }

  @Name("epsilon.InputCollection")
  @Label("Measured input collection")
  @Category("Epsilon")
  @Enabled(false)
  @StackTrace(false)
  public static final class Collection extends Event implements AutoCloseable {
    public String series;
    public int games;
    public int warmupGames;
    public boolean success;

    @Override
    public void close() {
      end();
      commit();
    }
  }
}
