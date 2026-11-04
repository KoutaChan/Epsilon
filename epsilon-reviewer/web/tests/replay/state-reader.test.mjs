import test from "node:test";
import assert from "node:assert/strict";
import { decodeResult } from "../../src/features/replay/model/decodeResult.ts";
import {
  CACHED_ROUNDS,
  createStateReader,
  maximumPlayerRowWidth,
} from "../../src/features/replay/model/stateReader.ts";
import { measurePlayerRowWidth } from "../../src/features/replay/renderer/tileLayout.ts";
import {
  discard,
  event,
  player,
  resultDocument,
  roundFromTables,
  table,
} from "./fixtures.mjs";

const NAMES = ["Player 1", "Player 2", "Player 3", "Player 4"];

function recordedTables(count = 100) {
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
    if (index % 3 === 0) river.push({ tile: "1p", tsumogiri: index % 2 === 0 });
    const players = [0, 1, 2, 3].map((seat) =>
      player(
        seat === 0
          ? {
              score: index < 5 ? 25000 : 24000 + index,
              riichi: index >= 5 && index < 12,
              hand: index % 2 ? ["2m", "3m", "5mr"] : ["1m", "2m", "3m"],
              draw: index % 2 ? null : "5pr",
              river: river.map((tile, position) =>
                discard(tile.tile, {
                  tsumogiri: tile.tsumogiri,
                  called: position === 0 && index >= 8,
                }),
              ),
              melds: index < 10 ? [] : [index < 11 ? pon : kan],
            }
          : {},
      ),
    );
    return table({
      players,
      remaining: 60 - (index % 60),
      activeSeat: index % 4,
      dora: index < 9 ? ["1m"] : ["1m", "9p"],
      kyotaku: index < 5 ? 0 : 1,
    });
  });
}

function recordedRound(tables, fields) {
  return roundFromTables(
    tables.map((entry, index) => ({
      step: index === 0 ? event("start_kyoku") : event("tsumo", { actor: 0 }),
      table: entry,
    })),
    fields,
  );
}

const named = (state) => ({
  ...state,
  players: state.players.map((entry, seat) => ({
    ...entry,
    seat,
    name: NAMES[seat],
  })),
});

test("start tables and splice deltas rebuild every recorded field after random seeks", () => {
  const tables = recordedTables();
  const document = JSON.parse(
    JSON.stringify(resultDocument([recordedRound(tables)])),
  );
  const unchanged = JSON.stringify(document);
  const model = decodeResult(document);
  assert.equal("players" in document.rounds[0].steps[1], false);
  assert.ok(document.rounds[0].steps.slice(1).every((step) => step.delta));
  const reader = createStateReader(model);
  const steps = model.rounds[0].steps;
  const order = [...tables.keys(), ...tables.keys()].map((_, index) =>
    index < tables.length
      ? index
      : ((index - tables.length) * 37) % tables.length,
  );
  for (const index of order)
    assert.deepEqual(reader.readSnapshot(0, steps[index]), {
      ...steps[index],
      ...named(tables[index]),
    });
  assert.equal(JSON.stringify(document), unchanged);
});

test("unchanged player objects and arrays are shared without changing earlier states", () => {
  const model = decodeResult(
    resultDocument([recordedRound(recordedTables(12))]),
  );
  const reader = createStateReader(model);
  const steps = model.rounds[0].steps;
  const first = reader.readSnapshot(0, steps[0]);
  const second = reader.readSnapshot(0, steps[1]);
  assert.equal(second.players[1], first.players[1]);
  assert.equal(second.players[0].river, first.players[0].river);
  assert.equal(second.players[0].melds, first.players[0].melds);
  assert.notEqual(second.players[0].hand, first.players[0].hand);
  assert.equal(first.players[0].draw, "5pr");
  assert.equal(second.players[0].draw, null);
  const changed = reader.readSnapshot(0, steps[8]);
  assert.equal(changed.players[0].river[0].called, true);
  assert.equal(first.players[0].river[0].called, undefined);
  assert.equal(reader.readSnapshot(0, steps[1]).players, second.players);
});

test("the reader keeps the most recent rounds and rebuilds an evicted one", () => {
  assert.equal(CACHED_ROUNDS, 3);
  const model = decodeResult(
    resultDocument(
      [0, 1, 2, 3].map((roundIndex) =>
        recordedRound(recordedTables(6), { roundIndex }),
      ),
    ),
  );
  const reads = [0, 0, 0, 0];
  const counted = {
    ...model,
    rounds: model.rounds.map((round, index) => ({
      ...round,
      get deltas() {
        reads[index]++;
        return round.deltas;
      },
    })),
  };
  const reader = createStateReader(counted);
  const read = (index) =>
    reader.readSnapshot(index, model.rounds[index].steps[5]);
  const first = read(0);
  read(0);
  read(1);
  read(2);
  assert.equal(read(0).players, first.players);
  assert.deepEqual(reads, [1, 1, 1, 0]);
  read(3);
  read(1);
  assert.deepEqual(reads, [1, 2, 1, 1]);
  read(2);
  assert.deepEqual(reads, [1, 2, 2, 1]);
  const rebuilt = read(0);
  assert.deepEqual(reads, [2, 2, 2, 1]);
  assert.notEqual(rebuilt.players, first.players);
  assert.deepEqual(rebuilt, first);
});

test("unselected rounds are not built and the camera width reads only melds", () => {
  const tables = recordedTables(40);
  const model = decodeResult(
    resultDocument([
      recordedRound(tables),
      recordedRound(recordedTables(40), { roundIndex: 1 }),
    ]),
  );
  for (const entry of model.rounds[1].start.players)
    Object.defineProperty(entry, "hand", {
      get() {
        throw new Error("Unselected hand built.");
      },
    });
  const reader = createStateReader(model);
  assert.equal(
    reader.readSnapshot(0, model.rounds[0].steps[1]).players[0].draw,
    null,
  );
  const expected =
    Math.max(
      measurePlayerRowWidth([]),
      ...tables.flatMap((entry) =>
        entry.players.map((value) => measurePlayerRowWidth(value.melds)),
      ),
    ) + 0.4;
  assert.equal(maximumPlayerRowWidth(model, measurePlayerRowWidth), expected);
  assert.ok(expected > measurePlayerRowWidth([]) + 0.4);
});
