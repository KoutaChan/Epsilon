export function playerSnapshot(seat, fields = {}) {
  return Object.freeze({
    seat,
    name: `Player ${seat + 1}`,
    score: 25000,
    riichi: false,
    hand: Object.freeze([]),
    draw: null,
    river: Object.freeze([]),
    melds: Object.freeze([]),
    ...fields,
  });
}

export function eventSnapshot(eventIndex, eventType, actor = -1, fields = {}) {
  return Object.freeze({
    index: eventIndex,
    eventIndex,
    stateId: eventIndex + 1,
    causeEventIndex: null,
    eventType,
    actor,
    decision: false,
    candidates: Object.freeze([]),
    players: Object.freeze([0, 1, 2, 3].map((seat) => playerSnapshot(seat))),
    dora: Object.freeze(["1m"]),
    honba: 0,
    kyotaku: 0,
    remaining: 60,
    activeSeat: 0,
    outcome:
      eventType === "hora"
        ? Object.freeze({
            type: "hora",
            actor,
            target: actor,
            winningTile: "1m",
            deltas: null,
            details: null,
          })
        : eventType === "ryukyoku"
          ? Object.freeze({
              type: "ryukyoku",
              actor: null,
              target: null,
              winningTile: null,
              deltas: null,
              details: null,
            })
          : null,
    ...fields,
  });
}

export function decisionSnapshot(cause, actor, type, fields = {}) {
  return Object.freeze({
    ...cause,
    actor,
    eventType: type,
    decision: true,
    causeEventIndex: cause.eventIndex,
    candidates: Object.freeze([
      Object.freeze({
        actionId: 0,
        type,
        tiles: [],
        probability: 1,
        chosen: true,
        tsumogiri: false,
      }),
    ]),
    outcome: null,
    ...fields,
  });
}

function arrayPatch(previous, current) {
  let index = 0;
  while (
    index < previous.length &&
    index < current.length &&
    JSON.stringify(previous[index]) === JSON.stringify(current[index])
  )
    index++;
  if (index === previous.length && index === current.length) return null;
  let suffix = 0;
  while (
    suffix < previous.length - index &&
    suffix < current.length - index &&
    JSON.stringify(previous.at(-1 - suffix)) ===
      JSON.stringify(current.at(-1 - suffix))
  )
    suffix++;
  return {
    index,
    removeCount: previous.length - index - suffix,
    values: current.slice(index, current.length - suffix),
  };
}
function tableState(step) {
  return {
    players: step.players.map(({ seat, name, ...player }) => player),
    dora: step.dora,
    honba: step.honba,
    kyotaku: step.kyotaku,
    remaining: step.remaining,
    activeSeat: step.activeSeat,
  };
}
function stateDelta(previous, current) {
  const players = [];
  for (let seat = 0; seat < 4; seat++) {
    const old = previous.players[seat],
      next = current.players[seat];
    const change = {
      seat,
      score: old.score === next.score ? null : next.score,
      riichi: old.riichi === next.riichi ? null : next.riichi,
      hand: arrayPatch(old.hand, next.hand),
      draw: old.draw === next.draw ? null : { tile: next.draw },
      river: arrayPatch(old.river, next.river),
      melds: arrayPatch(old.melds, next.melds),
    };
    if (
      Object.entries(change).some(
        ([key, value]) => key !== "seat" && value !== null,
      )
    )
      players.push(change);
  }
  return {
    players,
    dora: arrayPatch(previous.dora, current.dora),
    ...Object.fromEntries(
      ["honba", "kyotaku", "remaining", "activeSeat"].map((key) => [
        key,
        previous[key] === current[key] ? null : current[key],
      ]),
    ),
  };
}
export function reviewRound(steps) {
  const unique = new Map();
  for (const step of steps)
    if (!unique.has(step.stateId)) unique.set(step.stateId, tableState(step));
  const states = [...unique]
    .sort(([left], [right]) => left - right)
    .map(([stateId, checkpoint], ordinal, all) => ({
      stateId,
      checkpoint: ordinal % 32 === 0 ? checkpoint : null,
      delta:
        ordinal % 32 === 0 ? null : stateDelta(all[ordinal - 1][1], checkpoint),
    }));
  const thin = steps.map(
    ({ players, dora, honba, kyotaku, remaining, activeSeat, ...step }) => step,
  );
  return Object.freeze({
    id: "round-0",
    roundIndex: 0,
    honba: 0,
    dealer: 0,
    states: Object.freeze(states),
    steps: Object.freeze(thin),
  });
}
export function reviewResult(steps, fields = {}) {
  return Object.freeze({
    formatVersion: 4,
    resultId: "result-1",
    createdAt: "2026-09-15T00:00:00Z",
    metadata: Object.freeze({
      source: "mjai",
      names: ["Player 1", "Player 2", "Player 3", "Player 4"],
      finalScores: null,
      roundCount: 1,
    }),
    model: Object.freeze({
      modelId: "nano",
      revision: "test",
      displayName: "Nano",
      version: "1",
      series: "nano",
    }),
    rounds: Object.freeze([reviewRound(steps)]),
    ...fields,
  });
}
