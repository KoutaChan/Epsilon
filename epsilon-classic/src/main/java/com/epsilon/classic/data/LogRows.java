package com.epsilon.classic.data;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.grp.train.LogGames;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 牌譜を自己対局と同じエンジンで再生して（{@link ReplayDriver}）、判断ごとに行を作る。判断の記録 O は、卓と同じライター
 * （{@link ObservationWriter}）で作る。合法手が 1 つしかない判断と、選んだ手が牌譜から分からない応答（{@link ReplayDriver#UNOBSERVED}。
 * 優先度の高い手に消されたかもしれない）は行にしない。牌譜に記録の無い応答は、見送ったものとして行にする。検証用の半荘は牌譜 id の
 * ハッシュで決め、渡した割合の分だけ選ぶ（{@link #validation}）。
 *
 * <p>1 つのスレッドから使う。エンジンと食い違う牌譜では IllegalStateException を投げ、その半荘の行は足さない。
 */
public final class LogRows {

  private static final long VALIDATION_SALT = 0x7A1D_C3E5_9B24_6F08L;

  private final Variant variant;
  private final double validationShare;
  private final ReplayDriver driver;
  private final RuleWorkspace ws;
  private final ObservationWriter writer;
  private final Observation reader;

  /** 1 半荘分の O。行を足す前にここに溜める。 */
  private final ObservationBatch scratch;

  /** 1 半荘分の、行ごとの局の番号と選ばれた手。並びは O の番号とそろえる。 */
  private final List<int[]> pending = new ArrayList<>();

  /** @param validationShare 検証用にする半荘の割合（0 より大きく 1 より小さい） */
  public LogRows(Variant variant, double validationShare) {
    this.variant = Objects.requireNonNull(variant, "variant");
    if (!(validationShare > 0 && validationShare < 1)) {
      throw new IllegalArgumentException("The validation share must be in (0, 1): " + validationShare);
    }
    this.validationShare = validationShare;
    driver = new ReplayDriver(variant);
    ws = new RuleWorkspace(variant);
    writer = new ObservationWriter(variant);
    reader = new Observation(variant);
    scratch = new ObservationBatch(variant, 1 << 16);
  }

  /**
   * 半荘を検証用にするかを決める。牌譜 id のハッシュを [0, 1) の数にして、割合より小さければ検証用にする。割合を増やしても、
   * それまで検証用だった半荘は検証用のまま。
   *
   * @param recordId 牌譜 id
   * @param share 検証用にする半荘の割合
   * @return 検証用なら true
   */
  public static boolean validation(String recordId, double share) {
    return (SeedMixer.mix64(LogGames.gameId(recordId) ^ VALIDATION_SALT) >>> 11) * 0x1.0p-53 < share;
  }

  /**
   * 半荘を再生して、半荘の情報と判断ごとの行をビルダーに足す。
   *
   * @return 足した行の数
   * @throws IllegalArgumentException 牌譜かビルダーのルールが、コンストラクターで渡したルールと違うとき
   * @throws IllegalStateException 牌譜がエンジンと食い違うとき（このときは何も足さない）
   */
  public int add(RecordedGame game, ClassicRows.Builder into) {
    if (game.variant() != variant || into.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + game.variant());
    }
    scratch.clear();
    pending.clear();
    List<RecordedRound> rounds = game.rounds();
    List<RoundStart> starts = new ArrayList<>(rounds.size());
    for (int index = 0; index < rounds.size(); index++) {
      RecordedRound round = rounds.get(index);
      starts.add(round.start());
      Boundary boundary = driver.start(round, ws);
      Engine engine = driver.engine();
      while (boundary == Boundary.DECISIONS) {
        Decisions decisions = engine.decisions();
        for (int slot = 0; slot < decisions.size(); slot++) {
          int choice = driver.choice(slot);
          if (choice == ReplayDriver.UNOBSERVED || decisions.legalCount(slot) < 2) {
            continue;
          }
          writer.capture(engine.view(decisions.seat(slot)), decisions, slot, scratch);
          pending.add(new int[] {index, choice});
        }
        boundary = driver.step(ws);
      }
    }
    int gameIndex =
        into.game(
            new ClassicRows.Game(
                game.meta().id(), validation(game.meta().id(), validationShare), starts, game.finalScores()));
    for (int row = 0; row < pending.size(); row++) {
      int[] entry = pending.get(row);
      into.row(gameIndex, entry[0], entry[1], reader.bind(scratch, row));
    }
    return pending.size();
  }
}
