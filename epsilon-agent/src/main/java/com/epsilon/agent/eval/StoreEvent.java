package com.epsilon.agent.eval;

import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import java.util.Objects;

/**
 * 王者のストアに記録するイベント 1 行。最初の王者を除き、そのイベントを決めた決闘の名前（ジョブの名前）を持つ。ストアは 1 つの決闘に
 * つきイベントを 1 行だけ書く。
 */
@Choice(key = "kind")
public sealed interface StoreEvent {

  /** ストアが空のときに、決闘なしで置く最初の王者。 */
  @Case("INITIAL")
  record Initial(Champion champion) implements StoreEvent {

    public Initial {
      Objects.requireNonNull(champion, "champion");
    }
  }

  /** 決闘で「良い」が出て（確かめの卓があればそれにも通って）、王者を候補に替えた。前の王者は過去の王者になる。 */
  @Case("PROMOTED")
  record Promoted(Champion champion, String duel) implements StoreEvent {

    public Promoted {
      Objects.requireNonNull(champion, "champion");
      requireName(duel);
    }
  }

  /**
   * 決闘では「良い」が出たが、確かめの卓で外れたので王者を据え置いた。
   *
   * @param check 確かめの卓のジョブの名前
   */
  @Case("NOT_CONFIRMED")
  record NotConfirmed(Champion candidate, String duel, String check) implements StoreEvent {

    public NotConfirmed {
      Objects.requireNonNull(candidate, "candidate");
      requireName(duel);
      requireName(check);
    }
  }

  /** 王者との決闘で「悪い」が続いたので、run を止める。 */
  @Case("PAUSED")
  record Paused(String duel) implements StoreEvent {

    public Paused {
      requireName(duel);
    }
  }

  /** 王者との決闘で「悪い」が続いたので、次の反復を王者のネットワークから始める。王者は替えない。 */
  @Case("ROLLED_BACK")
  record RolledBack(Champion champion, String duel) implements StoreEvent {

    public RolledBack {
      Objects.requireNonNull(champion, "champion");
      requireName(duel);
    }
  }

  private static void requireName(String name) {
    if (name == null || name.isBlank()) {
      throw new IllegalArgumentException("A store event needs the name of its job");
    }
  }
}
