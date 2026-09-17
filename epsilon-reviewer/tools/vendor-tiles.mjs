import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createHash } from "node:crypto";

const commit = "26e127ba2117f45cdce5ea0225748cc0cfad3169";
const base = `https://raw.githubusercontent.com/FluffyStuff/riichi-mahjong-tiles/${commit}`;
const root = fileURLToPath(new URL("../", import.meta.url));
const destination = path.join(root, "web/public/tiles");
const names = ["Man", "Pin", "Sou"].flatMap((suit) => [
  ...Array.from({ length: 9 }, (_, i) => `${suit}${i + 1}`),
  `${suit}5-Dora`,
]);
names.push("Ton", "Nan", "Shaa", "Pei", "Haku", "Hatsu", "Chun", "Blank");
await mkdir(path.join(destination, "Regular"), { recursive: true });
const hashes = {};
const pending = [...names];
await Promise.all(
  Array.from({ length: 4 }, async () => {
    while (pending.length) {
      const name = pending.shift();
      const response = await fetch(`${base}/Regular/${name}.svg`);
      if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
      const bytes = Buffer.from(await response.arrayBuffer());
      if (!bytes.toString("utf8").includes("<svg"))
        throw new Error(`${name}: invalid SVG`);
      await writeFile(path.join(destination, "Regular", `${name}.svg`), bytes);
      hashes[`${name}.svg`] = createHash("sha256").update(bytes).digest("hex");
    }
  }),
);
for (const name of ["LICENSE.md", "README.md"]) {
  const response = await fetch(`${base}/${name}`);
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  await writeFile(path.join(destination, name), await response.text());
}
await writeFile(
  path.join(destination, "manifest.json"),
  JSON.stringify(
    {
      source: "FluffyStuff/riichi-mahjong-tiles",
      commit,
      license: "CC0-1.0",
      files: Object.fromEntries(Object.entries(hashes).sort()),
    },
    null,
    2,
  ) + "\n",
);
console.log(`Vendored ${names.length} original SVGs at ${commit}.`);
