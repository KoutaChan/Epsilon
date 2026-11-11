package com.epsilon.prism.learn;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.AuxiliaryLabels;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.rows.Row;
import java.util.Arrays;

/**
 * バッチの行（{@link Row}）から、補助のヘッドの教師（{@link AuxiliaryLabels} の形）を作る。エンコードのスレッドごとに 1 つ持ち、バッチごとに
 * {@link #clear} → 行ごとに {@link #add} → {@link #labels} の順に使う。
 *
 * <ul>
 *   <li>候補の教師: 測った手のうち、局末の記録があり、手の種類が候補の教師を持つ（{@link TargetSchema#hasCandidateTargets}）手。領域ごとの値は、
 *       {@link CandidateTarget} の宣言に従って列ごとに求める。局末の 3 ワードは {@link OutcomeCodec} で読み、局収支は局末の点 − 判断の時点の点。
 *   <li>読みの教師: Oracle の情報がある行の、相手の全ての席。牌の軸（待ち・放銃・放銃の点）は、直前にその行を書いたライターの色の入れ替えで
 *       並べ直し、入力の牌のトークンと同じ正規化した牌の順（そのルールの牌のトークンの番号）にする。
 * </ul>
 *
 * <p>相手の数と牌の軸の幅はルールで決まる（{@link TargetSchema#opponents}、牌のトークンの数）。
 *
 * <p>ヘッドの重みが 0 の群（候補か読み）は作らない（ネットワークもそのヘッドを計算しない）。
 */
public final class AuxiliaryTargets implements TargetWriter {

  private static final CandidateTarget[] CANDIDATE_TARGETS = CandidateTarget.values();
  private static final ReadingTarget[] READING_TARGETS = ReadingTarget.values();

  private final InputFormat input;
  private final int tiles;
  private final TargetSchema heads;
  private final OutcomeCodec outcomeCodec;
  private final OracleCodec oracleCodec;
  private final int opponents;
  private final int readingWidth;
  private final boolean candidates;
  private final boolean readings;
  private int[] candidateActions = new int[256];
  private int[] outcomes = new int[256];
  private float[] candidateValues;
  private int candidateCount;
  private int[] readingRows = new int[64];
  private float[] readingValues;
  private int readingCount;
  private final int[] shanten;
  private final long[] waits;
  private final long[] danger;
  private final float[] points;

  /**
   * 補助のヘッドの教師を作る準備をする。
   *
   * @param loss 補助のヘッドの損失（重みが正の群だけ教師を作る）
   * @param variant 行のルール（相手の数と牌の軸の幅が決まる）
   */
  public AuxiliaryTargets(AuxiliaryLoss loss, Variant variant) {
    input = InputFormat.of(variant);
    heads = TargetSchema.of(variant);
    outcomeCodec = OutcomeCodec.of(variant);
    oracleCodec = OracleCodec.of(variant);
    tiles = input.tiles();
    candidateValues = new float[256 * heads.candidateWidth()];
    opponents = heads.opponents();
    readingWidth = opponents * heads.readingWidth();
    readingValues = new float[64 * readingWidth];
    shanten = new int[opponents];
    waits = new long[opponents];
    danger = new long[opponents];
    points = new float[opponents * tiles];
    candidates = loss.resultWeight() > 0;
    readings = loss.oracleWeight() > 0;
  }

  @Override
  public void clear() {
    candidateCount = 0;
    readingCount = 0;
  }

  @Override
  public void add(
      int row, int firstAction, Row source, Observation o, InputFormat.RowWriter writer) {
    if (candidates) {
      addCandidates(firstAction, source, o);
    }
    if (readings && source.oracle().length > 0) {
      addReading(row, source.oracle(), writer);
    }
  }

  @Override
  public AuxiliaryLabels labels() {
    return new AuxiliaryLabels(
        heads,
        Arrays.copyOf(candidateActions, candidateCount),
        Arrays.copyOf(outcomes, candidateCount),
        Arrays.copyOf(candidateValues, candidateCount * heads.candidateWidth()),
        Arrays.copyOf(readingRows, readingCount),
        Arrays.copyOf(readingValues, readingCount * readingWidth));
  }

  private void addCandidates(int firstAction, Row source, Observation o) {
    long[] words = source.outcomes();
    int score = o.score(0);
    for (int action = 0, actions = source.actions(); action < actions; action++) {
      if (!source.measured(action)) {
        continue;
      }
      int word = action * OutcomeCodec.WORDS;
      Outcome outcome = outcomeCodec.outcome(words, word);
      if (outcome == null
          || !TargetSchema.hasCandidateTargets(input.role(o.action(action).type()))) {
        continue;
      }
      growCandidates();
      int base = candidateCount * heads.candidateWidth();
      for (CandidateTarget target : CANDIDATE_TARGETS) {
        for (int column = 0, width = heads.width(target); column < width; column++) {
          candidateValues[base + heads.offset(target) + column] =
              candidateValue(target, column, words, word, outcome, score);
        }
      }
      candidateActions[candidateCount] = firstAction + action;
      outcomes[candidateCount] = outcome.ordinal();
      candidateCount++;
    }
  }

  /** 候補の領域の 1 列の値（{@link AuxiliaryLabels} の契約どおりの、正規化の前の値）。 */
  private float candidateValue(
      CandidateTarget target, int column, long[] words, int word, Outcome outcome, int score) {
    return switch (target) {
      case OUTCOME -> flag(column == outcome.ordinal());
      case TENPAI -> flag(outcomeCodec.tenpai(words, word));
      case YAKU -> flag((outcomeCodec.yakuBits(words, word) >>> column & 1L) != 0L);
      case WIN_POINTS -> outcomeCodec.winPoints(words, word);
      case DEAL_IN_POINTS -> outcomeCodec.dealInPoints(words, word);
      case YAKU_TENPAI -> flag(outcomeCodec.yakuTenpai(words, word));
      case POINT_DELTA -> outcomeCodec.finalScore(words, word, 0) - score;
      case WINNER -> flag(column == outcomeCodec.winner(words, word) - 1);
    };
  }

  private void growCandidates() {
    if (candidateCount == candidateActions.length) {
      int capacity = candidateCount * 2;
      candidateActions = Arrays.copyOf(candidateActions, capacity);
      outcomes = Arrays.copyOf(outcomes, capacity);
      candidateValues = Arrays.copyOf(candidateValues, capacity * heads.candidateWidth());
    }
  }

  private void addReading(int row, long[] oracle, InputFormat.RowWriter writer) {
    // 放銃の点は、全ての相手の放銃の牌に通しで振った番号の順に並ぶ（相手の順、同じ相手の中では物理の牌の順）
    int pointIndex = 0;
    for (int index = 0; index < opponents; index++) {
      int opponent = index + 1;
      shanten[index] = oracleCodec.shanten(oracle, 0, opponent);
      waits[index] = canonical(oracleCodec.waits(oracle, 0, opponent), writer);
      long physical = oracleCodec.danger(oracle, 0, opponent);
      danger[index] = canonical(physical, writer);
      Arrays.fill(points, index * tiles, (index + 1) * tiles, 0f);
      for (long rest = physical; rest != 0L; rest &= rest - 1) {
        int tile = writer.canonicalTile(Long.numberOfTrailingZeros(rest));
        points[index * tiles + tile] = oracleCodec.points(oracle, 0, pointIndex++);
      }
    }
    if (readingCount == readingRows.length) {
      readingRows = Arrays.copyOf(readingRows, readingCount * 2);
      readingValues = Arrays.copyOf(readingValues, readingCount * 2 * readingWidth);
    }
    int base = readingCount * readingWidth;
    for (int index = 0; index < opponents; index++) {
      int unit = base + index * heads.readingWidth();
      for (ReadingTarget target : READING_TARGETS) {
        for (int column = 0, width = heads.width(target); column < width; column++) {
          readingValues[unit + heads.offset(target) + column] = readingValue(target, index, column);
        }
      }
    }
    readingRows[readingCount++] = row;
  }

  /** 読みの領域の 1 列の値（相手は {@code index}、牌の列は正規化した牌の順）。 */
  private float readingValue(ReadingTarget target, int index, int column) {
    return switch (target) {
      case TENPAI -> flag(shanten[index] == 0);
      case SHANTEN -> flag(column == TargetSchema.shantenClass(shanten[index]));
      case DANGER -> flag((danger[index] >>> column & 1L) != 0L);
      case WAITS -> flag((waits[index] >>> column & 1L) != 0L);
      case DANGER_POINTS -> points[index * tiles + column];
    };
  }

  /** 物理の牌のマスクを、ライターの色の入れ替えで、正規化した牌のトークンのマスクにする。 */
  private static long canonical(long physical, InputFormat.RowWriter writer) {
    long tiles = 0L;
    for (long rest = physical; rest != 0L; rest &= rest - 1) {
      tiles |= Tiles.bit(writer.canonicalTile(Long.numberOfTrailingZeros(rest)));
    }
    return tiles;
  }

  private static float flag(boolean value) {
    return value ? 1f : 0f;
  }
}
