package com.epsilon.grp;

/**
 * 局の終わり方から見た席の立場。OutcomeNetwork は、この局で立直していた確率を相対席と立場の組ごとに出す。終わり方が決まれば、
 * 席どうしは独立とみなす。荒牌で不聴の席は立直していないので、立直の項を持たない。
 */
public enum RiichiRole {
  /** 和了者（ダブロンの 2 人とも）。 */
  WINNER,
  /** ロンの放銃者。 */
  LOSER,
  /** ツモを払う席。 */
  PAYER,
  /** ロンに関わらない席。 */
  BYSTANDER,
  /** 荒牌で聴牌した席。 */
  TENPAI,
  /** 途中流局の席。 */
  ABORTIVE
}
