// メインプロセスと preload を、それぞれ 1 つの CommonJS のファイルにバンドルする。electron はバンドルに含めず、
// protocol の .ts は含める。
// 使い方: node desktop/scripts/build.mjs（どのフォルダーから実行してもよい）。
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
