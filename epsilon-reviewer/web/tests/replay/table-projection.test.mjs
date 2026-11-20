import { nukiTilePosition } from "../../src/features/replay/renderer/tileLayout.ts";
import {
  local,
  tableDimensions,
} from "../../src/features/replay/renderer/projection.ts";
import test from "node:test";
import assert from "node:assert/strict";

test("set-aside norths lie in the corner between the river and the next seat's river, apart from both", () => {
  for (const layout of ["portrait", "desktop", "landscape"]) {
    const d = tableDimensions(layout);
    const half = d.riverWidth / 2;
    const depth = (d.riverWidth * 2) / 3;
    const riverRight = d.riverStart + 5 * d.pitchX + half;
    // 下家（east）の河は、卓の座標で y が河の始まりの牌の手前の縁より奥にある。
    const nextRiverNear = local("east", d.riverStart - half, 0).y;
    const positions = [0, 1, 2, 3].map((index) => nukiTilePosition(index, d));
    assert.ok(positions[0].x - half > riverRight + d.riverWidth / 3, layout);
    for (const [index, { x, y }] of positions.entries()) {
      assert.equal(y, -d.riverRadius - 2 * d.pitchY, layout);
      if (index > 0)
        assert.ok(
          Math.abs(x - positions[index - 1].x - d.pitchX) < 1e-9,
          layout,
        );
      assert.ok(y + depth < nextRiverNear - d.riverWidth, layout);
      assert.ok(x + half < d.opponentRadius - depth, layout);
    }
  }
});
