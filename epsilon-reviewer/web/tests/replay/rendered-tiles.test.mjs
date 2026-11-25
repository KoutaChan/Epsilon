import test from "node:test";
import assert from "node:assert/strict";
import {
  concealedTileX,
  layoutPlayerRow,
  measurePlayerRowWidth,
  PLAYER_ROW_WIDTH,
} from "../../src/features/replay/renderer/tileLayout.ts";
import {
  local,
  tableDimensions,
} from "../../src/features/replay/renderer/projection.ts";
import { discard, meld } from "./fixtures.mjs";
import {
  EPSILON,
  MELD_ROWS,
  SCREEN_SIZES,
  SEATS,
  buildFrame,
  makeMeld,
  makeProjectionSnapshot,
  makeTileProjectionFixture,
} from "./table-fixtures.mjs";

test("rivers carry the riichi marker past called tiles without changing replay data", () => {
  const plain = discard("3s");
  const riichi = discard("4s", { riichi: true });
  const calledRiichi = discard("4s", { riichi: true, called: true });
  const calledDiscard = discard("3s", { called: true });
  const cases = [
    { name: "ordinary river", river: [plain, plain], sideways: -1 },
    { name: "declaration", river: [plain, riichi, plain], sideways: 1 },
    { name: "called declaration", river: [plain, calledRiichi], sideways: -1 },
    {
      name: "next discard",
      river: [plain, calledRiichi, plain, plain],
      sideways: 1,
    },
    {
      name: "successive calls",
      river: [calledDiscard, plain, calledRiichi, calledDiscard, plain, plain],
      sideways: 1,
    },
    {
      name: "end of row",
      river: [...Array(5).fill(plain), calledRiichi, plain, plain],
      sideways: 5,
    },
    {
      name: "start of next row",
      river: [...Array(6).fill(plain), calledRiichi, plain, plain],
      sideways: 6,
    },
    { name: "rewind before riichi", river: [plain, plain], sideways: -1 },
  ];
  const fixture = makeTileProjectionFixture();
  try {
    for (const mode of ["desktop", "landscape", "portrait"]) {
      const dimensions = tableDimensions(mode);
      for (const scenario of cases) {
        const river = Object.freeze(scenario.river);
        const snapshot = makeProjectionSnapshot([[], [], [], []], -1);
        for (const player of snapshot.players) player.river = river;
        fixture.layer.update({
          snapshot,
          event: { eventType: "dahai", actor: 0 },
          viewSeat: 0,
          reveal: true,
          layout: mode,
          labels: { draw: "Draw" },
          animateDraw: false,
        });
        for (const seat of SEATS) {
          const tiles = fixture.table.pieces.children.filter(
            (group) =>
              group.isGroup &&
              group.userData.kind === "river" &&
              group.userData.seat === seat,
          );
          const context = `${mode}: ${seat}: ${scenario.name}`;
          const baseAngle = local(seat, 0, 0).angle;
          assert.equal(
            tiles.length,
            river.filter((tile) => !tile.called).length,
            context,
          );
          for (const [index, tile] of tiles.entries()) {
            const sideways = index === scenario.sideways;
            assert.ok(
              Math.abs(
                tile.rotation.z - baseAngle - (sideways ? Math.PI / 2 : 0),
              ) < EPSILON,
              `${context}: discard ${index} has the wrong orientation`,
            );
            if (index % 6 === 0) continue;
            const previousWidth = index - 1 === scenario.sideways ? 4 / 3 : 1;
            const width = sideways ? 4 / 3 : 1;
            assert.ok(
              tile.position.distanceTo(tiles[index - 1].position) >=
                (dimensions.riverWidth * (previousWidth + width)) / 2 - EPSILON,
              `${context}: discard ${index} overlaps the previous tile`,
            );
          }
        }
      }
    }
  } finally {
    fixture.dispose();
  }
});

test("initial tile positions along every seat stay centered across record widths, viewpoint, draws and reveal settings", () => {
  const wideMelds = Array.from({ length: 4 }, () => makeMeld("DAIMINKAN", 3));
  const rowWidths = [
    measurePlayerRowWidth([]) + 0.4,
    measurePlayerRowWidth(wideMelds) + 0.4,
    PLAYER_ROW_WIDTH,
  ];
  const centeredRow = layoutPlayerRow([]);
  for (const rowWidth of rowWidths) {
    const fixture = makeTileProjectionFixture(rowWidth);
    try {
      for (const mode of ["desktop", "landscape", "portrait"])
        for (let viewSeat = 0; viewSeat < 4; viewSeat++)
          for (const drawnSeat of [-1, 0, 1, 2, 3])
            for (const reveal of [false, true]) {
              buildFrame(
                fixture,
                mode,
                [[], [], [], []],
                viewSeat,
                drawnSeat,
                reveal,
                false,
              );
              for (const [relativeSeat, seat] of SEATS.entries()) {
                const parent =
                  relativeSeat === 0
                    ? fixture.table.handGroup
                    : fixture.table.pieces;
                const hand = parent.children.filter(
                  (group) =>
                    group.isGroup &&
                    group.userData.kind === "hand" &&
                    group.userData.seat === seat,
                );
                const hasDraw = (viewSeat + relativeSeat) % 4 === drawnSeat;
                assert.equal(hand.length, 13 + Number(hasDraw));
                const axis = local(seat, 1, 0);
                for (const [index, group] of hand.entries()) {
                  // 卓の半径とカメラは、牌の列の幅（牌譜ごとに違う）に合わせて変わってよい。
                  const alongSeat =
                    (group.position.x * axis.x + group.position.y * axis.y) /
                    group.scale.x;
                  const expected = concealedTileX(
                    index,
                    index === 13,
                    centeredRow.handLeft,
                  );
                  assert.ok(
                    Math.abs(alongSeat - expected) < EPSILON,
                    `${mode}, row ${rowWidth}, view ${viewSeat}, ${seat}: an initial tile shifted along its seat`,
                  );
                }
              }
            }
    } finally {
      fixture.dispose();
    }
  }
});

test("viewed melds share the hand camera and scale with only added kans extending inward", () => {
  const fixture = makeTileProjectionFixture();
  try {
    for (const [width, height, mode] of SCREEN_SIZES) {
      fixture.fit(width, height, mode);
      for (const melds of MELD_ROWS) {
        buildFrame(
          fixture,
          mode,
          Array.from({ length: 4 }, () => melds),
          0,
          0,
          true,
          false,
        );
        const front = fixture.table.handGroup.children.filter((g) => g.isGroup);
        const hand = front.filter((g) => g.userData.kind === "hand");
        const exposed = front.filter((g) => g.userData.kind === "meld");
        assert.equal(hand.length, 14 - 3 * melds.length);
        assert.equal(
          exposed.length,
          melds.reduce((sum, meld) => sum + meld.tiles.length, 0),
        );
        const placements = layoutPlayerRow(melds).melds;
        for (const [index, group] of exposed.entries()) {
          assert.equal(group.userData.foreground, true);
          assert.equal(group.userData.seat, "south");
          assert.deepEqual(
            group.scale.toArray(),
            hand[0].scale.toArray(),
            "meld tiles must use the same physical scale as the hand",
          );
          assert.equal(
            group.position.y,
            hand[0].position.y + placements[index].y,
            "only the added tile may occupy the adjacent foreground row",
          );
        }
        const tableTiles = fixture.table.pieces.children.filter(
          (g) => g.isGroup,
        );
        assert.ok(
          tableTiles.every(
            (g) => g.userData.seat !== "south" || g.userData.kind === "river",
          ),
          "the viewed player must not retain a detached table meld row",
        );
        for (const seat of ["east", "north", "west"]) {
          const row = tableTiles.filter(
            (g) => g.userData.seat === seat && g.userData.kind !== "river",
          );
          const scale = tableDimensions(mode).opponentScale;
          let meldIndex = 0;
          for (const group of row) {
            assert.deepEqual(group.scale.toArray(), [scale, scale, scale]);
            const placement =
              group.userData.kind === "meld" ? placements[meldIndex++] : null;
            const radial =
              seat === "north" ? group.position.y : Math.abs(group.position.x);
            const expected =
              tableDimensions(mode).opponentRadius -
              (placement?.y ?? 0) * scale;
            assert.ok(
              Math.abs(radial - expected) < EPSILON,
              "only added kans may move inward from the concealed and called tile row",
            );
          }
        }
      }
    }
  } finally {
    fixture.dispose();
  }
});

test("only the seats whose player, last-discard mark or view changed are rebuilt", () => {
  const fixture = makeTileProjectionFixture();
  try {
    const snapshot = makeProjectionSnapshot([[], [], [], []], -1);
    const frame = {
      snapshot,
      event: { eventType: "dahai", actor: 1 },
      viewSeat: 0,
      reveal: true,
      layout: "desktop",
      labels: { draw: "Draw" },
      animateDraw: false,
    };
    assert.equal(fixture.layer.update(frame).rebuilt, true);
    const groups = () =>
      Object.fromEntries(
        SEATS.map((seat) => [
          seat,
          [
            ...fixture.table.pieces.children,
            ...fixture.table.handGroup.children,
          ].filter((group) => group.isGroup && group.userData.seat === seat),
        ]),
      );
    const before = groups();
    assert.equal(fixture.layer.update(frame).rebuilt, false);
    // 席 2 だけツモ牌が増えた局面。印は席 1 のまま。
    const drawn = {
      ...snapshot,
      players: snapshot.players.map((entry, seat) =>
        seat === 2 ? { ...entry, draw: "2p" } : entry,
      ),
    };
    const next = fixture.layer.update({
      ...frame,
      snapshot: drawn,
      event: { eventType: "tsumo", actor: 2 },
      animateDraw: true,
    });
    const after = groups();
    assert.equal(next.rebuilt, true);
    assert.equal(after.south[0], before.south[0]);
    assert.equal(after.west[0], before.west[0]);
    assert.notEqual(after.north[0], before.north[0]);
    // 席 1 の最後の打牌の印が外れるので、席 1 も作り直す。
    assert.notEqual(after.east[0], before.east[0]);
    assert.equal(
      next.arrival,
      after.north.filter((group) => group.userData.kind === "hand").at(-1),
    );
    assert.equal(next.annotations.length, 1);
    assert.equal(next.annotations[0].group, next.arrival);
  } finally {
    fixture.dispose();
  }
});

/** 点を含む多角形のうち、最後のもの（いちばん上に重なるもの）の位置を返す。凸の多角形だけを扱う。 */
