// メインプロセスと preload をそれぞれ 1 つの CommonJS にまとめる（electron は外。protocol の .ts は中へ）。
// 使い方: node desktop/scripts/build.mjs（どのフォルダーからでもよい）。
import { rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const desktop = fileURLToPath(new URL("..", import.meta.url));

await rm(new URL("../dist/", import.meta.url), {
  recursive: true,
  force: true,
});
await build({
  absWorkingDir: desktop,
  entryPoints: { main: "src/main.ts", preload: "src/preload.ts" },
  outdir: "dist",
  outExtension: { ".js": ".cjs" },
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "es2022",
  external: ["electron"],
  logLevel: "warning",
});
