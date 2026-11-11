package com.epsilon.prism.grp;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import java.nio.file.Path;
import java.util.Objects;

/** 自分の GRP の最初の項目 g000 の Φ。 */
@Choice(key = "kind", doc = "最初の項目の Φ の作り方")
public sealed interface PhiStartSettings {

  /** g000 の項目の種類。 */
  PhiEntryKind kind();

  /** 局末の点数の順位から始める。 */
  @Case("SCORE_RANK")
  record ScoreRank() implements PhiStartSettings {

    @Override
    public PhiEntryKind kind() {
      return PhiEntryKind.SCORE_RANK;
    }
  }

  /** 局末の点数のなめらかな関数から始める。 */
  @Case("SCORE_SMOOTH")
  record ScoreSmooth() implements PhiStartSettings {

    @Override
    public PhiEntryKind kind() {
      return PhiEntryKind.SCORE_SMOOTH;
    }
  }

  /**
   * 元の GRP から始める。LATE のモードは GrpNetwork を葉にして数え上げ、表を重ねられるのは GruSinkhorn だけなので、q と表は一緒に
   * 使えない。
   *
   * @param grp GRP のチェックポイント（grp.json と weights.f32 を直接含むディレクトリ）
   * @param table GRP が GruSinkhorn のときに重ねる PhiTable のファイル。無ければ重ねない
   * @param outcomes LATE のモードで最後の局を数え上げる、局の結果のネットワークのチェックポイント（grp.json と weights.f32 を直接含むディレクトリ）。無ければ
   *     g000 を使う間は LATE のモードにならない
   */
  @Case("GRP_CHECKPOINT")
  record GrpCheckpoint(
      @Key(doc = "GRP のチェックポイント。grp.json と weights.f32 を直接含むディレクトリ", presence = Presence.REQUIRED)
          Path grp,
      @Key(doc = "GRP が GruSinkhorn のときに重ねる PhiTable のファイル。無ければ重ねない", presence = Presence.ABSENT) Path table,
      @Key(doc = "LATE のモードで最後の局を数え上げる、局の結果のネットワークのチェックポイント。grp.json と weights.f32 を直接含むディレクトリ。表とは一緒に使えない",
              presence = Presence.ABSENT)
          Path outcomes)
      implements PhiStartSettings {

    public GrpCheckpoint {
      Objects.requireNonNull(grp, "grp");
      if (outcomes != null && table != null) {
        throw new InvalidSetting(
            "outcomes", "the late mode enumerates with a GrpNetwork leaf, and a Φ table overlays a GruSinkhorn only");
      }
    }

    @Override
    public PhiEntryKind kind() {
      return PhiEntryKind.NETWORK;
    }
  }
}
