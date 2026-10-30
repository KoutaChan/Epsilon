package com.epsilon.prism.learn;

import com.epsilon.core.Tile;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.Tokenizer;
import com.epsilon.prism.model.AuxiliaryLabels;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.rows.Row;
import java.util.Arrays;

/**
 * 塊の行（{@link Row}）から補助の頭の教師（{@link AuxiliaryLabels} の形）を作る。符号化のスレッドに 1 つ持ち、塊ごとに {@link #clear} →
 * 行ごとに {@link #add} → {@link #labels} の順に使う。
 *
 * <ul>
 *   <li>候補の教師: 測った手のうち、局末の記録があり、手の役割が {@link TargetSchema#hasCandidateTargets} の手。区画ごとの値は
 *       {@link CandidateTarget} の宣言から列ごとに引く（局末の 3 語を {@link OutcomeCodec} で解く。局収支は局末の点 − O の点）。
 *   <li>読みの教師: 神託のある行の相手 3 席。牌の軸（待ち・放銃・放銃の点）には、直前にその行を書いた {@link Tokenizer} の色の写しをかけ、
 *       入力の牌のトークンと同じ正準の牌の順にする。
 * </ul>
 *
 * <p>頭の重みが 0 の群（候補か読み）は作らない（網もその頭を計算しない）。
 */
public final class AuxiliaryTargets {

  private static final CandidateTarget[] CANDIDATE_TARGETS = CandidateTarget.values();
  private static final ReadingTarget[] READING_TARGETS = ReadingTarget.values();
  private static final int OPPONENTS = AuxiliaryLabels.OPPONENTS;
  private static final int TILES = Tile.NUM_TILE_TYPES;
  private static final int READING_WIDTH = OPPONENTS * ReadingTarget.WIDTH;

  private final boolean candidates;
  private final boolean readings;
  private int[] candidateActions = new int[256];
  private int[] outcomes = new int[256];
  private float[] candidateValues = new float[256 * CandidateTarget.WIDTH];
  private int candidateCount;
  private int[] readingRows = new int[64];
  private float[] readingValues = new float[64 * READING_WIDTH];
  private int readingCount;
  private final int[] shanten = new int[OPPONENTS];
  private final long[] waits = new long[OPPONENTS];
  private final long[] danger = new long[OPPONENTS];
  private final float[] points = new float[OPPONENTS * TILES];

  /**
   * 教師の作り手を作る。
   *
   * @param loss 補助の頭の損失（重みが正の群だけ教師を作る）
   */
  public AuxiliaryTargets(AuxiliaryLoss loss) {
    candidates = loss.resultWeight() > 0;
    readings = loss.oracleWeight() > 0;
  }

  /**
   * 補助の頭を計算するなら true を返す（どちらかの群の重みが正）。
   *
   * @param loss 補助の頭の損失
   * @return 補助の頭を計算するなら true
   */
  public static boolean needed(AuxiliaryLoss loss) {
    return loss.resultWeight() > 0 || loss.oracleWeight() > 0;
  }

  /** 塊の教師を捨てる。 */
  public void clear() {
    candidateCount = 0;
    readingCount = 0;
  }

  /**
   * 1 行の教師を足す。その行を {@link Tokenizer#write} した直後に、同じ O と同じ Tokenizer で呼ぶ。
   *
   * @param row 塊の中の行の番号
   * @param firstAction 塊の中のその行の最初の手の番号
   * @param source 行
   * @param o 行の記録 O（{@code source} の O に付けたもの）
   * @param tokenizer 直前にこの行を書いた Tokenizer（色の写し）
   */
  public void add(int row, int firstAction, Row source, Observation o, Tokenizer tokenizer) {
    if (candidates) {
      addCandidates(firstAction, source, o);
    }
    if (readings && source.oracle().length > 0) {
      addReading(row, source.oracle(), tokenizer);
    }
  }

  /**
   * 足した行の教師を返す（配列は写しで、作り手を使い回してよい）。
   *
   * @return 教師
   */
  public AuxiliaryLabels labels() {
    return new AuxiliaryLabels(
        Arrays.copyOf(candidateActions, candidateCount),
        Arrays.copyOf(outcomes, candidateCount),
        Arrays.copyOf(candidateValues, candidateCount * CandidateTarget.WIDTH),
        Arrays.copyOf(readingRows, readingCount),
        Arrays.copyOf(readingValues, readingCount * READING_WIDTH));
  }

  private void addCandidates(int firstAction, Row source, Observation o) {
    long[] words = source.outcomes();
    int score = o.score(0);
    for (int action = 0, actions = source.actions(); action < actions; action++) {
      if (!source.measured(action)) {
        continue;
      }
      int word = action * OutcomeCodec.WORDS;
      Outcome outcome = OutcomeCodec.outcome(words, word);
      if (outcome == null
          || !TargetSchema.hasCandidateTargets(ActionRole.of(o.action(action).type()))) {
        continue;
      }
      growCandidates();
      int base = candidateCount * CandidateTarget.WIDTH;
      for (CandidateTarget target : CANDIDATE_TARGETS) {
        for (int column = 0, width = target.width(); column < width; column++) {
          candidateValues[base + target.offset() + column] =
              candidateValue(target, column, words, word, outcome, score);
        }
      }
      candidateActions[candidateCount] = firstAction + action;
      outcomes[candidateCount] = outcome.ordinal();
      candidateCount++;
    }
  }

  /** 候補の区画の列の値（{@link AuxiliaryLabels} の約束の生の値）。 */
  private static float candidateValue(
      CandidateTarget target, int column, long[] words, int word, Outcome outcome, int score) {
    return switch (target) {
      case OUTCOME -> flag(column == outcome.ordinal());
      case TENPAI -> flag(OutcomeCodec.tenpai(words, word));
      case YAKU -> flag((OutcomeCodec.yakuBits(words, word) >>> column & 1L) != 0L);
      case WIN_POINTS -> OutcomeCodec.winPoints(words, word);
      case DEAL_IN_POINTS -> OutcomeCodec.dealInPoints(words, word);
      case YAKU_TENPAI -> flag(OutcomeCodec.yakuTenpai(words, word));
      case POINT_DELTA -> OutcomeCodec.finalScore(words, word, 0) - score;
      case WINNER -> flag(column == OutcomeCodec.winner(words, word) - 1);
    };
  }

  private void growCandidates() {
    if (candidateCount == candidateActions.length) {
      int capacity = candidateCount * 2;
      candidateActions = Arrays.copyOf(candidateActions, capacity);
      outcomes = Arrays.copyOf(outcomes, capacity);
      candidateValues = Arrays.copyOf(candidateValues, capacity * CandidateTarget.WIDTH);
    }
  }

  private void addReading(int row, long[] oracle, Tokenizer tokenizer) {
    // 放銃の点は、相手の順・相手の中は物理の牌の順に、全ての相手の放銃の牌を通しで数えた番号で並ぶ
    int pointIndex = 0;
    for (int index = 0; index < OPPONENTS; index++) {
      int opponent = index + 1;
      shanten[index] = OracleCodec.shanten(oracle, 0, opponent);
      waits[index] = canonical(OracleCodec.waits(oracle, 0, opponent), tokenizer);
      long physical = OracleCodec.danger(oracle, 0, opponent);
      danger[index] = canonical(physical, tokenizer);
      Arrays.fill(points, index * TILES, (index + 1) * TILES, 0f);
      for (long rest = physical; rest != 0L; rest &= rest - 1) {
        int tile = tokenizer.canonicalTile(Long.numberOfTrailingZeros(rest));
        points[index * TILES + tile] = OracleCodec.points(oracle, 0, pointIndex++);
      }
    }
    if (readingCount == readingRows.length) {
      readingRows = Arrays.copyOf(readingRows, readingCount * 2);
      readingValues = Arrays.copyOf(readingValues, readingCount * 2 * READING_WIDTH);
    }
    int base = readingCount * READING_WIDTH;
    for (int index = 0; index < OPPONENTS; index++) {
      int unit = base + index * ReadingTarget.WIDTH;
      for (ReadingTarget target : READING_TARGETS) {
        for (int column = 0, width = target.width(); column < width; column++) {
          readingValues[unit + target.offset() + column] = readingValue(target, index, column);
        }
      }
    }
    readingRows[readingCount++] = row;
  }

  /** 読みの区画の列の値（相手 {@code index}、牌の列は正準の牌）。 */
  private float readingValue(ReadingTarget target, int index, int column) {
    return switch (target) {
      case TENPAI -> flag(shanten[index] == 0);
      case SHANTEN -> flag(column == TargetSchema.shantenClass(shanten[index]));
      case DANGER -> flag((danger[index] >>> column & 1L) != 0L);
      case WAITS -> flag((waits[index] >>> column & 1L) != 0L);
      case DANGER_POINTS -> points[index * TILES + column];
    };
  }

  /** 物理の牌のマスクを、Tokenizer の色の写しで正準の牌のマスクにする。 */
  private static long canonical(long physical, Tokenizer tokenizer) {
    long tiles = 0L;
    for (long rest = physical; rest != 0L; rest &= rest - 1) {
      tiles |= 1L << tokenizer.canonicalTile(Long.numberOfTrailingZeros(rest));
    }
    return tiles;
  }

  private static float flag(boolean value) {
    return value ? 1f : 0f;
  }
}
