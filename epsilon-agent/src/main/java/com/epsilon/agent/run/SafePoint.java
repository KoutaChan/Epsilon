package com.epsilon.agent.run;

import java.util.Objects;

/**
 * セーフポイント。止める・起動し直す・一時停止する・GPU を譲るのは、ここでだけ行う。セーフポイントで止まった run は、もう一度起動すれば
 * 止めずに回したときと同じ結果になる。
 *
 * @param place セーフポイントの場所
 * @param name 段かジョブの名前（場所が段やジョブに依らなければ空）
 */
public record SafePoint(Place place, String name) {

  /** セーフポイントの場所。 */
  public enum Place {
    /** 起動の直後、リソースを開く前。 */
    LAUNCH,
    /** 段の前。 */
    BEFORE_STEP,
    /** 段の後。 */
    AFTER_STEP,
    /** 段の中で待っている間（Φ の待ちなど）。 */
    WAITING,
    /** 反復を閉じた後、その反復のジョブを回す前。 */
    ITERATION,
    /** ジョブの前。 */
    BEFORE_JOB,
    /** ジョブの中で、1 チャンクを終えた後。 */
    CHUNK,
    /** ジョブの後。 */
    AFTER_JOB,
    /** 待っているジョブを全て終えた後、次の反復の前。 */
    JOBS
  }

  public SafePoint {
    Objects.requireNonNull(place, "place");
    Objects.requireNonNull(name, "name");
  }

  public static SafePoint launch() {
    return new SafePoint(Place.LAUNCH, "");
  }

  public static SafePoint beforeStep(String step) {
    return new SafePoint(Place.BEFORE_STEP, step);
  }

  public static SafePoint afterStep(String step) {
    return new SafePoint(Place.AFTER_STEP, step);
  }

  public static SafePoint waiting(String step) {
    return new SafePoint(Place.WAITING, step);
  }

  public static SafePoint iteration() {
    return new SafePoint(Place.ITERATION, "");
  }

  public static SafePoint beforeJob(String job) {
    return new SafePoint(Place.BEFORE_JOB, job);
  }

  public static SafePoint chunk(String job) {
    return new SafePoint(Place.CHUNK, job);
  }

  public static SafePoint afterJob(String job) {
    return new SafePoint(Place.AFTER_JOB, job);
  }

  public static SafePoint jobs() {
    return new SafePoint(Place.JOBS, "");
  }

  /** 停止と再起動の要求が、このセーフポイントで効くか。 */
  public boolean reaches(Request.At at) {
    return switch (at) {
      case STEP -> true;
      case ITERATION -> place == Place.ITERATION;
      case JOBS -> place == Place.JOBS;
    };
  }

  /** 一時停止の要求が、このセーフポイントで効くか。 */
  public boolean pauses(Request.Pause pause) {
    return pause.before() == null || (place == Place.BEFORE_STEP && pause.before().equals(name));
  }

  @Override
  public String toString() {
    return name.isEmpty() ? place.toString() : place + " " + name;
  }
}
