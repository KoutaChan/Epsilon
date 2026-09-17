import test from "node:test";
import assert from "node:assert/strict";
import {
  createReplayStateReader,
  maximumPlayerRowWidth,
} from "../src/features/replay/stateReader.ts";
import { readCurrentReviewResult } from "../src/api/reviewFormat.ts";
import { buildPlaybackPositions } from "../src/features/replay/timeline.ts";
import { ApiError } from "../src/api/errors.ts";
import { measurePlayerRowWidth } from "../src/features/replay/renderer/tileLayout.js";
import {
  eventSnapshot,
  decisionSnapshot,
  playerSnapshot,
  reviewResult,
} from "./reviewFixtures.mjs";

function recordedStates(count = 100) {
  const river = [];
  const pon = {
    type: "PON",
    tiles: ["5mr", "5m", "5m"],
    from: 3,
    calledIndex: 0,
    addedIndex: -1,
  };
  const kan = {
    ...pon,
    type: "KAKAN",
    tiles: [...pon.tiles, "5m"],
    addedIndex: 3,
  };
  return Array.from({ length: count }, (_, index) => {
    if (index % 3 === 0)
      river.push({
        tile: "1p",
        tsumogiri: index % 2 === 0,
        riichi: false,
        called: false,
      });
    const players = [0, 1, 2, 3].map((seat) =>
      playerSnapshot(
        seat,
        seat === 0
          ? {
              score: index < 5 ? 25000 : 24000 + index,
              riichi: index >= 5 && index < 12,
              hand: index % 2 ? ["2m", "3m", "5mr"] : ["1m", "2m", "3m"],
              draw: index % 2 ? null : "5pr",
              river: river.map((tile, position) => ({
                ...tile,
                called: position === 0 && index >= 8,
              })),
              melds: index < 10 ? [] : [index < 11 ? pon : kan],
            }
          : {},
      ),
    );
    return eventSnapshot(index, "tsumo", 0, {
      players,
      stateId: 7 + index * 3,
      remaining: 60 - (index % 60),
      activeSeat: index % 4,
      dora: index < 9 ? ["1m"] : ["1m", "9p"],
      kyotaku: index < 5 ? 0 : 1,
      candidates: [
        {
          actionId: 41,
          type: "DAHAI",
          tiles: ["5mr"],
          probability: 0.12345678901234567,
          chosen: true,
          tsumogiri: false,
        },
      ],
    });
  });
}

test("v4 checkpoints and sparse state patches reconstruct every recorded field after random seeks", () => {
  const source = recordedStates();
  const wire = JSON.parse(JSON.stringify(reviewResult(source)));
  const unchanged = JSON.stringify(wire);
  assert.equal(readCurrentReviewResult(wire), wire);
  assert.deepEqual(
    wire.rounds[0].states.flatMap((entry, index) =>
      entry.checkpoint ? [index] : [],
    ),
    [0, 32, 64, 96],
  );
  assert.equal("players" in wire.rounds[0].steps[0], false);
  assert.equal("name" in wire.rounds[0].states[0].checkpoint.players[0], false);
  const reader = createReplayStateReader(wire);
  const order = [...source.keys(), ...source.keys()].map((_, index) =>
    index < source.length
      ? index
      : ((index - source.length) * 37) % source.length,
  );
  for (const index of order)
    assert.deepEqual(
      reader.readSnapshot(0, wire.rounds[0].steps[index]),
      source[index],
    );
  assert.equal(JSON.stringify(wire), unchanged);
});

test("unchanged player objects and arrays are shared without changing earlier states", () => {
  const wire = reviewResult(recordedStates(12));
  const reader = createReplayStateReader(wire);
  const first = reader.readState(0, wire.rounds[0].states[0].stateId);
  const second = reader.readState(0, wire.rounds[0].states[1].stateId);
  assert.equal(second.players[1], first.players[1]);
  assert.equal(second.players[0].river, first.players[0].river);
  assert.equal(second.players[0].melds, first.players[0].melds);
  assert.notEqual(second.players[0].hand, first.players[0].hand);
  assert.equal(first.players[0].draw, "5pr");
  assert.equal(second.players[0].draw, null);
  const changed = reader.readState(0, wire.rounds[0].states[8].stateId);
  assert.equal(changed.players[0].river[0].called, true);
  assert.equal(first.players[0].river[0].called, false);
});

test("a random seek reads one checkpoint and at most its thirty-one following deltas", () => {
  const result = reviewResult(recordedStates(100));
  const accessed = [];
  const states = result.rounds[0].states.map((entry, index) => ({
    stateId: entry.stateId,
    get checkpoint() {
      accessed.push(["checkpoint", index]);
      return entry.checkpoint;
    },
    get delta() {
      accessed.push(["delta", index]);
      return entry.delta;
    },
  }));
  const wire = { ...result, rounds: [{ ...result.rounds[0], states }] };
  const reader = createReplayStateReader(wire);
  assert.deepEqual(accessed, []);
  reader.readState(0, states[63].stateId);
  assert.deepEqual(accessed, [
    ["checkpoint", 32],
    ...Array.from({ length: 31 }, (_, index) => ["delta", index + 33]),
  ]);
  accessed.length = 0;
  reader.readState(0, states[39].stateId);
  assert.deepEqual(accessed, []);
});

test("state cache evicts old chunks and can reconstruct them for backward navigation", () => {
  const result = reviewResult(recordedStates(170));
  const reads = new Map();
  const states = result.rounds[0].states.map((entry, index) => ({
    ...entry,
    get checkpoint() {
      reads.set(index, (reads.get(index) ?? 0) + 1);
      return entry.checkpoint;
    },
  }));
  const reader = createReplayStateReader({
    ...result,
    rounds: [{ ...result.rounds[0], states }],
  });
  const first = reader.readState(0, states[0].stateId);
  for (const index of [32, 64, 96, 0, 128, 0])
    reader.readState(0, states[index].stateId);
  assert.equal(reads.get(0), 1);
  reader.readState(0, states[32].stateId);
  assert.equal(reads.get(32), 2);
  assert.deepEqual(reader.readState(0, states[0].stateId), first);
});

test("unselected rounds are not decoded and camera width reads only melds", () => {
  const source = recordedStates(40),
    result = reviewResult(source);
  const future = {
    ...reviewResult(source).rounds[0],
    id: "round-1",
    roundIndex: 1,
  };
  const wire = { ...result, rounds: [result.rounds[0], future] };
  for (const entry of future.states)
    if (entry.checkpoint) {
      for (const player of entry.checkpoint.players)
        Object.defineProperty(player, "hand", {
          get() {
            throw new Error("Unselected hand decoded.");
          },
        });
    }
  const reader = createReplayStateReader(wire);
  assert.equal(reader.readState(0, source[1].stateId).players[0].draw, null);
  const expected =
    Math.max(
      measurePlayerRowWidth([]),
      ...source.flatMap((step) =>
        step.players.map((player) => measurePlayerRowWidth(player.melds)),
      ),
    ) + 0.4;
  assert.equal(maximumPlayerRowWidth(wire, measurePlayerRowWidth), expected);
});

test("all candidates and unknown versus empty scoring metadata retain their exact values", () => {
  const start = eventSnapshot(0, "start_kyoku");
  const candidates = Array.from({ length: 8 }, (_, actionId) => ({
    actionId,
    type: "DAHAI",
    tiles: ["1m"],
    probability: (actionId + 1) / 36,
    chosen: actionId === 7,
    tsumogiri: actionId === 0,
  }));
  const details = {
    han: 3,
    fu: 40,
    points: 5200,
    yakuman: 0,
    yaku: [{ code: "URADORA", han: 2, yakuman: 0 }],
    uraIndicators: [],
  };
  const win = eventSnapshot(1, "hora", 0, {
    candidates,
    outcome: {
      type: "hora",
      actor: 0,
      target: 1,
      winningTile: "5mr",
      deltas: [5200, -5200, 0, 0],
      details,
    },
  });
  const result = reviewResult([start, win]);
  const decoded = createReplayStateReader(result).readSnapshot(
    0,
    result.rounds[0].steps[1],
  );
  assert.equal(decoded.candidates, candidates);
  assert.equal(decoded.candidates.length, 8);
  assert.equal(decoded.outcome.details, details);
  assert.deepEqual(decoded.outcome.details.uraIndicators, []);
});

for (const mutation of [
  (result) => {
    result.rounds[0].steps[0].stateId = 9999;
  },
  (result) => {
    result.rounds[0].states[1].stateId = result.rounds[0].states[0].stateId;
  },
  (result) => {
    result.rounds[0].states[1].checkpoint =
      result.rounds[0].states[0].checkpoint;
  },
  (result) => {
    result.rounds[0].states[32].checkpoint = null;
  },
  (result) => {
    result.rounds[0].states[1].delta.players[0].hand.index = 1000;
  },
  (result) => {
    result.rounds[0].states[1].delta.players[0].hand.removeCount = -1;
  },
  (result) => {
    result.rounds[0].states[1].delta.players[0].draw = {};
  },
  (result) => {
    result.rounds[0].states[1].delta.players.push(
      result.rounds[0].states[1].delta.players[0],
    );
  },
])
  test(`malformed state references and patches are rejected at the boundary: ${mutation}`, () => {
    const wire = JSON.parse(JSON.stringify(reviewResult(recordedStates(40))));
    mutation(wire);
    assert.throws(
      () => readCurrentReviewResult(wire),
      (error) => error instanceof ApiError && error.code === "invalid_result",
    );
  });

test("snapshot wrappers are released with their state chunk", () => {
  const result = reviewResult(recordedStates(160));
  const reader = createReplayStateReader(result);
  const steps = result.rounds[0].steps;
  const first = reader.readSnapshot(0, steps[0]);
  assert.equal(reader.readSnapshot(0, steps[0]), first);
  for (const ordinal of [32, 64, 96, 128])
    reader.readSnapshot(0, steps[ordinal]);
  const restored = reader.readSnapshot(0, steps[0]);
  assert.notEqual(restored, first);
  assert.deepEqual(restored, first);
});

test("thin decision links display the recorded state across dora updates and multiple ron outcomes", () => {
  const start = eventSnapshot(0, "start_kyoku");
  const draw = eventSnapshot(1, "tsumo", 0);
  const dora = eventSnapshot(2, "dora", -1, { dora: ["1m", "2p"] });
  const discardChoice = decisionSnapshot(draw, 0, "DAHAI", {
    eventIndex: 3,
    stateId: dora.stateId,
    dora: dora.dora,
  });
  const discard = eventSnapshot(3, "dahai", 0, { dora: dora.dora });
  const ronA = decisionSnapshot(discard, 1, "RON_AGARI", { eventIndex: 4 });
  const ronB = decisionSnapshot(discard, 2, "RON_AGARI", { eventIndex: 5 });
  const winA = eventSnapshot(4, "hora", 1, { dora: dora.dora });
  const winB = eventSnapshot(5, "hora", 2, { dora: dora.dora });
  const result = reviewResult([
    start,
    draw,
    dora,
    discardChoice,
    discard,
    ronA,
    ronB,
    winA,
    winB,
  ]);
  readCurrentReviewResult(result);
  const reader = createReplayStateReader(result);
  const positions = buildPlaybackPositions(result.rounds[0].steps, true);
  assert.deepEqual(
    positions.map((position) => position.kind),
    ["table", "table", "table", "round-result", "match-result"],
  );
  assert.equal("players" in positions[1].snapshot, false);
  assert.deepEqual(reader.readSnapshot(0, positions[1].snapshot).dora, [
    "1m",
    "2p",
  ]);
  assert.deepEqual(reader.readSnapshot(0, positions[1].event).dora, ["1m"]);
  assert.deepEqual(
    positions[2].decisions.map((step) => step.actor),
    [1, 2],
  );
  assert.deepEqual(
    positions[3].outcomeEvents.map(
      (step) => reader.readSnapshot(0, step).outcome.actor,
    ),
    [1, 2],
  );
});
