package com.epsilon.prism.grp;

/** 関門に合格した候補の入れ方（設定 {@code epsilon.prism.grpSwitch}）。 */
public enum GrpSwitch {

  /** 候補を refits.jsonl に READY で公開するだけで、入れるのは手動の grp-switch。 */
  PROPOSE,

  /** コホート h までで作った候補を反復 h + grpSwitchLag で入れる（{@link AutoSwitch}）。その時に READY でなければ入れない。 */
  AUTO
}
