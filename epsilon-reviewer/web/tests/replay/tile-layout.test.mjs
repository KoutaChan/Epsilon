import { tileCopies } from "@epsilon-reviewer/protocol/tiles";
import { displaySeats } from "../../src/features/replay/model/seats.ts";
import test from "node:test";
import assert from "node:assert/strict";
import {
  concealedTileX,
  layoutPlayerRow,
  measurePlayerRowWidth,
  PLAYER_ROW_CENTER,
  PLAYER_ROW_WIDTH,
} from "../../src/features/replay/renderer/tileLayout.ts";
import {
  local,
  tableDimensions,
} from "../../src/features/replay/renderer/projection.ts";
import { meld } from "./fixtures.mjs";
import { EPSILON, MELD_ROWS, SEATS, makeMeld } from "./table-fixtures.mjs";

function tileBounds(seat, x, y, width, height, placement) {
  const corners = [-width / 2, width / 2].flatMap((dx) =>
    [-height / 2, height / 2].map((dy) => local(seat, x + dx, y + dy)),
  );
  return {
    minX: Math.min(...corners.map((p) => p.x)),
    maxX: Math.max(...corners.map((p) => p.x)),
    minY: Math.min(...corners.map((p) => p.y)),
    maxY: Math.max(...corners.map((p) => p.y)),
    seat,
    placement,
  };
}

function boundsOverlap(a, b) {
  return (
    a.minX < b.maxX - EPSILON &&
    a.maxX > b.minX + EPSILON &&
    a.minY < b.maxY - EPSILON &&
    a.maxY > b.minY + EPSILON
  );
}

function assertSeparatedTiles(tiles, label) {
  for (let i = 0; i < tiles.length; i++)
    for (let j = i + 1; j < tiles.length; j++)
      assert.equal(
        boundsOverlap(tiles[i], tiles[j]),
        false,
        `${label}: tiles ${i} and ${j} overlap`,
      );
}

function meldFootprints(row, seat, scale = 1, radius = 0) {
  return row.melds.map((p) =>
    tileBounds(
      seat,
      p.x * scale,
      p.y * scale - radius,
      (p.sideways ? 4 / 3 : 1) * scale,
      (p.sideways ? 1 : 4 / 3) * scale,
      p,
    ),
  );
}

function handFootprints(
  row,
  meldCount,
  seat,
  standing,
  phase,
  scale = 1,
  radius = 0,
) {
  const count = 13 - 3 * meldCount,
    length = count + (phase === "before-draw" ? 0 : 1);
  return Array.from({ length }, (_, index) =>
    tileBounds(
      seat,
      concealedTileX(
        index,
        phase === "after-draw" && index === count,
        row.handLeft,
      ) * scale,
      -radius,
      scale,
      (standing ? 0.61 : 4 / 3) * scale,
    ),
  );
}

function riverFootprints(seat, dimensions) {
  const w = dimensions.riverWidth;
  return Array.from({ length: 18 }, (_, index) => {
    const riichi = index === 17;
    return tileBounds(
      seat,
      dimensions.riverStart +
        (index % 6) * dimensions.pitchX +
        (riichi ? w / 6 : 0),
      -dimensions.riverRadius - Math.floor(index / 6) * dimensions.pitchY,
      riichi ? (w * 4) / 3 : w,
      riichi ? w : (w * 4) / 3,
    );
  });
}

test("concealed tiles and base melds stay in one bounded row with added kans adjacent through draw and call phases", () => {
  const left = PLAYER_ROW_CENTER - PLAYER_ROW_WIDTH / 2;
  const right = PLAYER_ROW_CENTER + PLAYER_ROW_WIDTH / 2;
  for (const melds of MELD_ROWS) {
    const row = layoutPlayerRow(melds),
      exposed = meldFootprints(row, "south");
    assert.ok(row.width > 0 && row.width <= PLAYER_ROW_WIDTH);
    for (const phase of [
      "before-draw",
      "after-draw",
      ...(melds.length ? ["after-call"] : []),
    ]) {
      const hand = handFootprints(row, melds.length, "south", true, phase);
      const tiles = [...hand, ...exposed];
      assertSeparatedTiles(tiles, `${melds.map((m) => m.type)} ${phase}`);
      for (const tile of tiles) {
        assert.ok(
          tile.minX >= left - EPSILON,
          `${phase}: tile exceeds the left row envelope`,
        );
        assert.ok(
          tile.maxX <= right + EPSILON,
          `${phase}: tile exceeds the right row envelope`,
        );
        assert.ok(
          Math.abs((tile.minY + tile.maxY) / 2 - (tile.placement?.y ?? 0)) <
            EPSILON,
          "only an added kan tile may move toward the center from the base row",
        );
      }
      if (exposed.length) {
        const gap =
          Math.min(...exposed.map((tile) => tile.minX)) -
          Math.max(...hand.map((tile) => tile.maxX));
        assert.ok(
          gap >= 0.5 - EPSILON,
          `${phase}: only ${gap} units separate the draw and meld`,
        );
      }
      if (phase === "after-draw") {
        const gap = hand.at(-1).minX - hand.at(-2).maxX;
        assert.ok(
          gap > 0.25 && gap < 0.75,
          "the draw must remain adjacent but visibly separated",
        );
      }
    }
  }
});

test("meld placement preserves every physical tile with added kans beside their called tiles", () => {
  for (const melds of MELD_ROWS) {
    const row = layoutPlayerRow(melds);
    assert.equal(
      row.melds.length,
      melds.reduce((sum, meld) => sum + meld.tiles.length, 0),
    );
    assertSeparatedTiles(meldFootprints(row, "south"), "one-row meld groups");
    for (const meld of melds) {
      const own = row.melds.filter((p) => p.meld === meld);
      assert.deepEqual(
        own.map((p) => p.index).sort(),
        meld.tiles.map((_, index) => index),
      );
      for (const placement of own)
        assert.equal(
          placement.sideways,
          placement.index === meld.calledIndex || placement.added,
        );
      const base = own.filter((placement) => !placement.added);
      for (let index = 1; index < base.length; index++)
        assert.ok(
          base[index].x > base[index - 1].x,
          "reversing meld groups must preserve the tile order inside each group",
        );
      const added = own.find((p) => p.added);
      if (meld.type === "KAKAN") {
        const called = own.find((p) => p.index === meld.calledIndex);
        assert.equal(added.x, called.x);
        assert.equal(added.index, meld.addedIndex);
        const gap = added.y - called.y - 1;
        assert.ok(
          gap > 0 && gap < 0.08,
          "the two sideways tiles need a narrow visible gap",
        );
      } else assert.equal(added, undefined);
      for (const placement of own)
        if (!placement.added) assert.equal(placement.y, 0);
    }
    for (let index = 1; index < melds.length; index++) {
      const previous = meldFootprints(
        { melds: row.melds.filter((p) => p.meld === melds[index - 1]) },
        "south",
      );
      const next = meldFootprints(
        { melds: row.melds.filter((p) => p.meld === melds[index]) },
        "south",
      );
      assert.ok(
        Math.max(...next.map((p) => p.maxX)) <
          Math.min(...previous.map((p) => p.minX)),
        "each newer meld must remain separately readable to the left",
      );
    }
  }
});

test("initial hands center their reserved draw slot independently of future meld widths", () => {
  const wideMelds = Array.from({ length: 4 }, () => makeMeld("DAIMINKAN", 3));
  const rowWidths = [
    measurePlayerRowWidth([]) + 0.4,
    measurePlayerRowWidth(wideMelds) + 0.4,
    PLAYER_ROW_WIDTH,
  ];
  const initial = layoutPlayerRow([], rowWidths[0]);
  for (const rowWidth of rowWidths) {
    const row = layoutPlayerRow([], rowWidth);
    assert.equal(
      row.handLeft,
      initial.handLeft,
      "future meld space must not shift the initial hand",
    );
    const hand = handFootprints(row, 0, "south", true, "after-draw");
    const left = hand[0].minX,
      right = hand.at(-1).maxX;
    assert.ok(
      Math.abs((left + right) / 2 - PLAYER_ROW_CENTER) < EPSILON,
      "the thirteen tiles and reserved draw slot must be centered together",
    );
    assert.ok(left >= PLAYER_ROW_CENTER - rowWidth / 2 - EPSILON);
    assert.ok(right <= PLAYER_ROW_CENTER + rowWidth / 2 + EPSILON);
  }
});

test("appending one through four mixed melds leaves older groups fixed at the row's right edge", () => {
  const melds = Object.freeze([
    makeMeld("PON", 2),
    makeMeld("CHI", 0),
    makeMeld("ANKAN"),
    makeMeld("DAIMINKAN", 1),
  ]);
  const measuredWidth = measurePlayerRowWidth(melds) + 0.4;
  for (const rowWidth of [PLAYER_ROW_WIDTH, measuredWidth]) {
    const right = PLAYER_ROW_CENTER + rowWidth / 2;
    let previous = layoutPlayerRow([], rowWidth);
    for (let count = 1; count <= melds.length; count++) {
      const row = layoutPlayerRow(melds.slice(0, count), rowWidth);
      assert.deepEqual(
        row.melds.filter((placement) => placement.meld !== melds[count - 1]),
        previous.melds,
        "a newly called group must not move or rotate any older tile",
      );
      const bounds = meldFootprints(row, "south");
      assert.ok(
        Math.abs(Math.max(...bounds.map((tile) => tile.maxX)) - right) <
          EPSILON,
        "the oldest group must remain anchored at the fixed right edge",
      );
      if (count > 1) {
        const added = bounds.filter(
          (tile) => tile.placement.meld === melds[count - 1],
        );
        const older = bounds.filter(
          (tile) => tile.placement.meld !== melds[count - 1],
        );
        assert.ok(
          Math.max(...added.map((tile) => tile.maxX)) <
            Math.min(...older.map((tile) => tile.minX)),
          "the new group must occupy space to the left of every older group",
        );
      }
      previous = row;
    }
  }
});

test("upgrading a pon within multiple melds preserves every existing tile and the hand origin", () => {
  for (const calledIndex of [0, 1, 2]) {
    for (const ponIndex of [0, 1, 2, 3]) {
      const pon = makeMeld("PON", calledIndex);
      const kan = makeMeld("KAKAN", calledIndex);
      const melds = [makeMeld("CHI", 0), makeMeld("ANKAN"), makeMeld("PON", 1)];
      melds.splice(ponIndex, 0, pon);
      const upgraded = melds.map((meld) => (meld === pon ? kan : meld));
      for (const rowWidth of [
        PLAYER_ROW_WIDTH,
        measurePlayerRowWidth(melds) + 0.4,
      ]) {
        const before = layoutPlayerRow(melds, rowWidth);
        const after = layoutPlayerRow(upgraded, rowWidth);
        assert.equal(
          after.handLeft,
          before.handLeft,
          "adding a kan tile moved the concealed hand",
        );
        assert.equal(
          after.width,
          before.width,
          "adding a kan tile widened the base row",
        );
        for (let groupIndex = 0; groupIndex < melds.length; groupIndex++) {
          const oldTiles = before.melds.filter(
            (tile) => tile.meld === melds[groupIndex],
          );
          const newTiles = after.melds.filter(
            (tile) => tile.meld === upgraded[groupIndex] && !tile.added,
          );
          assert.equal(newTiles.length, oldTiles.length);
          for (let tileIndex = 0; tileIndex < oldTiles.length; tileIndex++) {
            const oldTile = oldTiles[tileIndex],
              newTile = newTiles[tileIndex];
            assert.equal(
              newTile.index,
              oldTile.index,
              "a base tile changed its display order",
            );
            assert.equal(
              newTile.x,
              oldTile.x,
              "a base tile moved horizontally during added kan",
            );
            assert.equal(
              newTile.y,
              oldTile.y,
              "a base tile moved vertically during added kan",
            );
            assert.equal(
              newTile.sideways,
              oldTile.sideways,
              "added kan changed the original call direction",
            );
          }
        }
        const added = after.melds.find((tile) => tile.added);
        const called = after.melds.find(
          (tile) => tile.meld === kan && tile.index === calledIndex,
        );
        assert.equal(added.x, called.x);
        assert.ok(
          added.y > called.y,
          "the added tile must stay beside its original called tile",
        );
      }
    }
  }
});

test("combined rows of every seat stay clear of neighboring hands and full rivers", () => {
  for (const mode of ["desktop", "landscape", "portrait"]) {
    const d = tableDimensions(mode),
      rivers = SEATS.flatMap((seat) => riverFootprints(seat, d));
    for (const melds of MELD_ROWS) {
      const row = layoutPlayerRow(melds);
      for (const standing of [false, true]) {
        const rows = SEATS.flatMap((seat) => [
          ...handFootprints(
            row,
            melds.length,
            seat,
            standing,
            "after-draw",
            d.opponentScale,
            d.opponentRadius,
          ),
          ...meldFootprints(row, seat, d.opponentScale, d.opponentRadius),
        ]);
        assertSeparatedTiles(rows, `${mode}: neighboring combined player rows`);
        for (const tile of rows)
          for (const river of rivers)
            assert.equal(
              boundsOverlap(tile, river),
              false,
              `${mode}: ${tile.seat} row enters a river`,
            );
      }
    }
  }
});

test("three players leave the opposite seat empty and have no 2m to 8m", () => {
  assert.deepEqual(displaySeats(4), [0, 1, 2, 3]);
  assert.deepEqual(displaySeats(3), [0, 1, null, 2]);
  assert.equal(tileCopies("5m", 3), 0);
  assert.equal(tileCopies("5mr", 3), 0);
  assert.equal(tileCopies("9m", 3), 4);
  assert.equal(tileCopies("5pr", 3), 1);
  assert.equal(tileCopies("5m"), 3);
});

test("only real melds shorten the hand: a pon moves the centred hand by one and a half tiles", () => {
  const open = layoutPlayerRow([meld("PON", ["P", "P", "P"])]);
  assert.equal(open.melds.length, 3);
  assert.ok(
    Math.abs(open.handLeft - layoutPlayerRow([]).handLeft - 1.5 * 1.012) < 1e-9,
  );
  assert.ok(
    measurePlayerRowWidth([meld("PON", ["P", "P", "P"])]) >
      measurePlayerRowWidth([]),
  );
});
