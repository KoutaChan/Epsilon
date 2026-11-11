package com.epsilon.agent.run;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import java.util.Objects;

/**
 * run を制御する要求。1 件を番号付きのファイル 1 つとして run の control に置き、段とジョブを回す JVM がセーフポイントで読む。GRP の
 * 学び直しの要求だけは、同じ JVM の GRP の裏の作業が読む。
 */
@Choice(key = "kind")
public sealed interface Request {

  /** 停止と再起動の要求が効く場所。 */
  enum At {
    /** 次のセーフポイント。 */
    STEP,
    /** 反復を閉じた後、その反復のジョブを回す前。 */
    ITERATION,
    /** その反復のジョブを全て終えた後。停止の要求だけが使う。 */
    JOBS
  }

  /** ファイル名に使う要求の種類（stop・restart・pause・resume・cancel・grp-refit）。 */
  default String label() {
    return switch (this) {
      case Stop stop -> "stop";
      case Restart restart -> "restart";
      case Pause pause -> "pause";
      case Resume resume -> "resume";
      case Cancel cancel -> "cancel";
      case GrpRefit refit -> "grp-refit";
    };
  }

  /** 止める。もう一度起動すれば、止めた所から続ける。 */
  @Case("STOP")
  record Stop(At at) implements Request {

    public Stop {
      Objects.requireNonNull(at, "at");
    }
  }

  /** 起動し直す（次の起動から効くキーの変更を効かせる）。 */
  @Case("RESTART")
  record Restart(At at) implements Request {

    public Restart {
      Objects.requireNonNull(at, "at");
      if (at == At.JOBS) {
        throw new IllegalArgumentException("A restart takes effect at STEP or ITERATION");
      }
    }
  }

  /**
   * 一時停止。再開の要求で消されるまで、セーフポイントで待つ。
   *
   * @param before この名前の段の前でだけ待つ。無ければ次のセーフポイントで待つ
   */
  @Case("PAUSE")
  record Pause(@Key(doc = "この名前の段の前でだけ待つ", presence = Presence.ABSENT) String before) implements Request {}

  /** 再開。これより前に置いた一時停止の要求を消す。 */
  @Case("RESUME")
  record Resume() implements Request {}

  /**
   * まだ従っていない要求を 1 つ取り消す。
   *
   * @param request 取り消す要求の番号
   */
  @Case("CANCEL")
  record Cancel(@Range(min = 1) int request) implements Request {}

  /**
   * GRP の学び直し。自分の GRP の run で、学習の JVM の GRP の裏の作業が、時点のコホートが確定した後に 1 回だけ従う。段と仕事を回す仕組みは
   * この頼みに従わず、取り消しの頼みが指したときだけ消す。
   *
   * @param slot 学び直す時点（このコホートまでの半荘の記録で学ぶ）
   */
  @Case("GRP_REFIT")
  record GrpRefit(@Range(min = 1) int slot) implements Request {}
}
