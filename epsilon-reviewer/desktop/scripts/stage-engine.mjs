// desktop/engine を組む。electron-builder が resources/engine へそのまま写す（asar の外。Java は asar を読めない）。
// jar と実行環境は先に作っておき、ここでは写しと確かめだけをする。
//
//   node desktop/scripts/stage-engine.mjs --runtime <dir> --jar <file> --models <models.json>
//     [--libtorch <dir>] [--jni <jar>] [--out <dir>]
//
// --runtime  Windows の JDK の jmods から jlink で作った Windows x64 の実行環境（bin/java.exe）。
// --jar      backend の desktop の jar（mvnw -pl epsilon-reviewer/backend -am package -DskipTests
//            "-P!linux-native,desktop" が作る epsilon-reviewer-1.0-SNAPSHOT-desktop.jar）。
// --models   入れるモデルの models.json（backend/models.example.json と同じ形。checkpoint はこのファイルの
//            フォルダーからの相対か絶対）。系列ごとの推論のファイルだけを写す。
// --libtorch libtorch 2.11.0 CPU の DLL（既定 %USERPROFILE%\.djl.ai\pytorch\2.11.0-cpu-win-x86_64）。
// --jni      djl_torch.dll を取り出す jar（既定 ~/.m2 の pytorch-jni-cpu-win-x86_64-2.11.0-0.40.4-rocm-SNAPSHOT.jar）。
// --out      既定 desktop/engine。
//
// 7 つの DLL が scripts/native-files.json の大きさと SHA-256 に合わなければ止まる。その後に desktop で
// npx electron-builder --dir（web の dist を先に作る）。
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, cp, mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { inflateRawSync } from "node:zlib";

const NATIVE_FILES = JSON.parse(
  await readFile(new URL("native-files.json", import.meta.url), "utf8"),
);
const JNI_FILE = "djl_torch.dll";
const JNI_ENTRY = "jnilib/win-x86_64/cpu/djl_torch.dll";
const DECISION_FILES = ["manifest.json", "architecture.id", "decision-0000.params"];
/** 系列ごとの推論のファイル（学習の副ファイル campaign-state.json などは入れない）。 */
const INFERENCE_FILES = {
  "epsilon-prism": ["policy.json", "m0/model.json", "m0/weights.bin"],
  "epsilon-nano": DECISION_FILES,
  "epsilon-pico": DECISION_FILES,
  epsilon: DECISION_FILES,
};
const MODEL_FIELDS = ["modelId", "revision", "displayName", "version", "series", "checkpoint"];
const PATH_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

const { values: options } = parseArgs({
  options: {
    runtime: { type: "string" },
    jar: { type: "string" },
    models: { type: "string" },
    libtorch: {
      type: "string",
      default: path.join(os.homedir(), ".djl.ai", "pytorch", "2.11.0-cpu-win-x86_64"),
    },
    jni: {
      type: "string",
      default: path.join(
        os.homedir(),
        ".m2/repository/ai/djl/pytorch/pytorch-jni-cpu-win-x86_64/2.11.0-0.40.4-rocm-SNAPSHOT",
        "pytorch-jni-cpu-win-x86_64-2.11.0-0.40.4-rocm-SNAPSHOT.jar",
      ),
    },
    out: { type: "string", default: fileURLToPath(new URL("../engine", import.meta.url)) },
  },
});
for (const name of ["runtime", "jar", "models"])
  if (options[name] === undefined) throw new Error(`--${name} is required.`);

const out = path.resolve(options.out);
await rm(out, { recursive: true, force: true });
await cp(options.runtime, path.join(out, "runtime"), { recursive: true });
await stat(path.join(out, "runtime", "bin", "java.exe"));
await mkdir(path.join(out, "lib"), { recursive: true });
await copyFile(options.jar, path.join(out, "lib", "epsilon-reviewer.jar"));
await stageNative(path.join(out, "native"));
await stageModels(path.resolve(options.models), path.join(out, "models"));
for (const part of ["runtime", "lib", "native", "models"])
  console.log(`${part.padEnd(8)}${(await treeSize(path.join(out, part))).toLocaleString("en-US")} B`);

/** 6 つは libtorch のフォルダーから、djl_torch.dll は JNI の jar から。全て固定の大きさと SHA-256 で確かめる。 */
async function stageNative(target) {
  await mkdir(target);
  for (const { name, size, sha256 } of NATIVE_FILES) {
    const file = path.join(target, name);
    if (name === JNI_FILE) await writeFile(file, zipEntry(await readFile(options.jni), JNI_ENTRY));
    else await copyFile(path.join(options.libtorch, name), file);
    const actual = { size: (await stat(file)).size, sha256: await fileDigest(file) };
    if (actual.size !== size || actual.sha256 !== sha256)
      throw new Error(`${name} does not match native-files.json (${actual.size} B, ${actual.sha256}).`);
  }
}

/** 推論のファイルを models/<modelId>/<revision>/ へ写し、相対パス・CPU・sha256 の models.json を書く。 */
async function stageModels(source, target) {
  const catalog = JSON.parse(await readFile(source, "utf8"));
  const models = [];
  for (const entry of catalog.models ?? []) {
    for (const field of MODEL_FIELDS)
      if (typeof entry[field] !== "string" || entry[field] === "")
        throw new Error(`models[].${field} must be a non-empty string.`);
    const files = INFERENCE_FILES[entry.series];
    if (files === undefined) throw new Error(`Unknown series: ${entry.series}.`);
    if (!PATH_SEGMENT.test(entry.modelId) || !PATH_SEGMENT.test(entry.revision))
      throw new Error(`The model ID and revision must be folder names: ${entry.modelId}/${entry.revision}.`);
    const checkpoint = `${entry.modelId}/${entry.revision}`;
    if (models.some((model) => model.checkpoint === checkpoint))
      throw new Error(`Duplicate model: ${checkpoint}.`);
    const from = path.resolve(path.dirname(source), entry.checkpoint);
    const to = path.join(target, entry.modelId, entry.revision);
    for (const file of files) {
      await mkdir(path.dirname(path.join(to, file)), { recursive: true });
      await copyFile(path.join(from, file), path.join(to, file));
    }
    const { modelId, revision, displayName, version, series } = entry;
    models.push({
      modelId,
      revision,
      displayName,
      version,
      series,
      checkpoint,
      options: { device: "cpu" },
      sha256: await checkpointDigest(to),
    });
  }
  const first = models[0];
  if (first === undefined) throw new Error("models.json lists no models.");
  const { modelId, revision } = catalog.defaultModel ?? first;
  if (!models.some((model) => model.modelId === modelId && model.revision === revision))
    throw new Error(`defaultModel ${modelId}/${revision} is not in models.`);
  await writeFile(
    path.join(target, "models.json"),
    JSON.stringify({ models, defaultModel: { modelId, revision } }, null, 2) + "\n",
  );
}

/**
 * チェックポイントのフォルダーの SHA-256（Java の CheckpointDigest と同じ）: 普通のファイルを
 * "/" でつないだ相対パスの UTF-8 のバイト順に、パス・0x00・大きさ（8 バイト BE）・中身。
 */
async function checkpointDigest(directory) {
  const names = (await readdir(directory, { recursive: true, withFileTypes: true }))
    .filter((entry) => entry.isFile())
    .map((entry) =>
      path.relative(directory, path.join(entry.parentPath, entry.name)).split(path.sep).join("/"),
    )
    .sort((left, right) => Buffer.compare(Buffer.from(left), Buffer.from(right)));
  const hash = createHash("sha256");
  for (const name of names) {
    const file = path.join(directory, name);
    const size = Buffer.alloc(8);
    size.writeBigUInt64BE(BigInt((await stat(file)).size));
    hash.update(Buffer.from(name, "utf8")).update(Buffer.of(0)).update(size);
    for await (const chunk of createReadStream(file)) hash.update(chunk);
  }
  return hash.digest("hex");
}

async function fileDigest(file) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest("hex");
}

/** zip（jar）の 1 項目を取り出す。ZIP64 でない小さな jar だけを読む。 */
function zipEntry(zip, name) {
  const end = zip.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  if (end < 0) throw new Error("The JNI jar is not a zip file.");
  let offset = zip.readUInt32LE(end + 16);
  for (let count = zip.readUInt16LE(end + 10); count > 0; count--) {
    const nameLength = zip.readUInt16LE(offset + 28);
    if (zip.toString("utf8", offset + 46, offset + 46 + nameLength) === name) {
      const method = zip.readUInt16LE(offset + 10);
      const local = zip.readUInt32LE(offset + 42);
      const start = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
      const raw = zip.subarray(start, start + zip.readUInt32LE(offset + 20));
      if (method === 0) return raw;
      if (method === 8) return inflateRawSync(raw);
      throw new Error(`Unsupported zip method ${method} for ${name}.`);
    }
    offset += 46 + nameLength + zip.readUInt16LE(offset + 30) + zip.readUInt16LE(offset + 32);
  }
  throw new Error(`${name} is not in the JNI jar.`);
}

async function treeSize(directory) {
  let total = 0;
  for (const entry of await readdir(directory, { recursive: true, withFileTypes: true }))
    if (entry.isFile()) total += (await stat(path.join(entry.parentPath, entry.name))).size;
  return total;
}
