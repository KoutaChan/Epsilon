// 解析結果の文書を試験のために組み立てる。局面は局の始めの全体と、手順ごとの Splice の差分で書く。

const HONORS = ["E", "S", "W", "N", "P", "F", "C"];
const RED_FIVES = ["5mr", "5pr", "5sr"];

export function player(fields = {}) {
  return {
    score: 25000,
    riichi: false,
    hand: [],
    draw: null,
    river: [],
    melds: [],
    ...fields,
  };
}

export function table(fields = {}) {
  return {
    players: [0, 1, 2, 3].map(() => player()),
    dora: ["1m"],
    honba: 0,
    kyotaku: 0,
    remaining: 60,
    activeSeat: 0,
    ...fields,
  };
}

/** 局面の 1 席（stateReader が作る形。名前と席を持つ）。 */
export function playerSnapshot(seat, fields = {}) {
  return { ...player(fields), seat, name: `Player ${seat + 1}` };
}

/** 出来事の手順とその局面（Snapshot の形）。局の結果は hora と ryukyoku だけが持つ。 */
export function snapshot(eventType, actor = -1, fields = {}) {
  return {
    index: 0,
    decision: false,
    eventIndex: 0,
    eventType,
    actor,
    frame: 0,
    outcome:
      eventType === "hora"
        ? hora(actor, actor)
        : eventType === "ryukyoku"
          ? { type: "ryukyoku" }
          : null,
    players: [0, 1, 2, 3].map((seat) => playerSnapshot(seat)),
    dora: ["1m"],
    honba: 0,
    kyotaku: 0,
    remaining: 60,
    activeSeat: 0,
    ...fields,
  };
}

/** 河の 1 枚。印は真のときだけ書く。 */
export function discard(tile, marks = {}) {
  const value = { tile };
  for (const mark of ["tsumogiri", "riichi", "called"])
    if (marks[mark]) value[mark] = true;
  return value;
}

export function event(kind, fields = {}) {
  return { event: kind, ...fields };
}

export function decision(seat, candidates, fields = {}) {
  return { seat, candidates, ...fields };
}

export function hora(actor, target, winningTile = "1m", details) {
  return {
    type: "hora",
    actor,
    target,
    winningTile,
    ...(details === undefined ? {} : { details }),
  };
}

export function round(steps, fields = {}) {
  return {
    roundIndex: 0,
    honba: 0,
    dealer: 0,
    firstEvent: 0,
    start: table(),
    steps,
    ...fields,
  };
}

export function resultDocument(rounds, fields = {}) {
  return {
    format: 6,
    resultId: "00000000-0000-4000-8000-000000000001",
    createdAt: "2026-11-04T00:00:00.000Z",
    model: {
      modelId: "prism",
      revision: "test",
      displayName: "Prism",
      version: "1",
    },
    record: {
      source: "mjai",
      names: ["Player 1", "Player 2", "Player 3", "Player 4"],
      roundCount: rounds.length,
    },
    rounds,
    ...fields,
  };
}

/** 局面の並びから局を作る。entries は {step, table} で、各手順の差分を前の局面から作る。 */
export function roundFromTables(entries, fields = {}) {
  return round(
    entries.map(({ step, table: current }, index) => {
      if (index === 0) return step;
      const delta = stateDelta(entries[index - 1].table, current);
      return delta === null ? step : { ...step, delta };
    }),
    { start: entries[0].table, ...fields },
  );
}

const same = (left, right) => JSON.stringify(left) === JSON.stringify(right);

/** 前の配列から今の配列への 1 か所の置き換え。同じなら null。 */
export function splice(previous, current) {
  let index = 0;
  while (
    index < previous.length &&
    index < current.length &&
    same(previous[index], current[index])
  )
    index++;
  if (index === previous.length && index === current.length) return null;
  let suffix = 0;
  while (
    suffix < previous.length - index &&
    suffix < current.length - index &&
    same(previous.at(-1 - suffix), current.at(-1 - suffix))
  )
    suffix++;
  return [
    index,
    previous.length - index - suffix,
    ...current.slice(index, current.length - suffix),
  ];
}

export function stateDelta(previous, current) {
  const delta = {};
  const players = [];
  for (let seat = 0; seat < 4; seat++) {
    const before = previous.players[seat],
      after = current.players[seat];
    const change = {};
    if (before.score !== after.score) change.score = after.score;
    if (before.riichi !== after.riichi) change.riichi = after.riichi;
    for (const field of ["hand", "river", "melds"]) {
      const patch = splice(before[field], after[field]);
      if (patch) change[field] = patch;
    }
    if (before.draw !== after.draw) change.draw = after.draw;
    if (Object.keys(change).length > 0) players.push({ seat, ...change });
  }
  if (players.length > 0) delta.players = players;
  const dora = splice(previous.dora, current.dora);
  if (dora) delta.dora = dora;
  for (const field of ["honba", "kyotaku", "remaining", "activeSeat"])
    if (previous[field] !== current[field]) delta[field] = current[field];
  return Object.keys(delta).length > 0 ? delta : null;
}

function tileType(tile) {
  const honor = HONORS.indexOf(tile);
  return honor >= 0
    ? 27 + honor
    : "mps".indexOf(tile[1]) * 9 + Number(tile[0]) - 1;
}

/** 打牌の行動 ID（Action.java の配置）。riichi はリーチの打牌。 */
export function discardAction(
  tile,
  { tsumogiri = false, riichi = false } = {},
) {
  const red = RED_FIVES.indexOf(tile);
  const offset =
    red >= 0
      ? 68 + red + (tsumogiri ? 3 : 0)
      : tileType(tile) + (tsumogiri ? 34 : 0);
  return (riichi ? 294 : 0) + offset;
}

/** 牌を書かない手と鳴きの行動 ID。鳴きは variant で別の ID にする。 */
export const ACTIONS = {
  chi: (variant = 0) => 74 + variant,
  pon: (variant = 0) => 155 + variant,
  daiminkan: (variant = 0) => 192 + variant,
  ankan: (variant = 0) => 226 + variant,
  kakan: (variant = 0) => 260 + variant,
  tsumo: 368,
  ron: 369,
  pass: 370,
  kyushu: 371,
};
