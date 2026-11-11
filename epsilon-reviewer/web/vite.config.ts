import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";
import { brotliCompress, constants, gzip } from "node:zlib";
import { defineConfig, type Plugin } from "vite";

// サーバが返す Content-Security-Policy ヘッダーから、frame-ancestors を除いたもの（frame-ancestors は meta では効かない）。
const CONTENT_SECURITY_POLICY =
  "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'self'";
const API_TARGET = process.env.EPSILON_REVIEWER_API ?? "http://127.0.0.1:8080";
const LOCAL_HOST = /^(localhost|127\.0\.0\.1)(:\d+)?$/;
const COMPRESSIBLE = /\.(?:js|css|html|svg|json)$/;
/** この大きさ（バイト）以下のファイルは、圧縮版を作らない。 */
const COMPRESS_THRESHOLD = 1024;

const compressBrotli = promisify(brotliCompress);
const compressGzip = promisify(gzip);

/** ビルドした index.html の head の先頭に CSP の meta を入れる。 */
function contentSecurityPolicy(): Plugin {
  return {
    name: "epsilon-content-security-policy",
    apply: "build",
    transformIndexHtml: () => [
      {
        tag: "meta",
        attrs: {
          "http-equiv": "Content-Security-Policy",
          content: CONTENT_SECURITY_POLICY,
        },
        injectTo: "head-prepend",
      },
    ],
  };
}

/**
 * dist の下にある js・css・html・svg・json のうち 1 KiB を超えるファイルごとに、同じ場所へ .br（品質 11）と
 * .gz（レベル 9）を書く。サーバは起動時にこれらを読む。
 */
function precompress(): Plugin {
  return {
    name: "epsilon-precompress",
    apply: "build",
    async writeBundle({ dir }) {
      if (dir === undefined) return;
      const paths = await readdir(dir, { recursive: true });
      await Promise.all(
        paths
          .filter((path) => COMPRESSIBLE.test(path))
          .map(async (path) => {
            const file = join(dir, path);
            const source = await readFile(file);
            if (source.length <= COMPRESS_THRESHOLD) return;
            const [br, gz] = await Promise.all([
              compressBrotli(source, {
                params: {
                  [constants.BROTLI_PARAM_MODE]: constants.BROTLI_MODE_TEXT,
                  [constants.BROTLI_PARAM_QUALITY]:
                    constants.BROTLI_MAX_QUALITY,
                  [constants.BROTLI_PARAM_SIZE_HINT]: source.length,
                },
              }),
              compressGzip(source, { level: constants.Z_BEST_COMPRESSION }),
            ]);
            await Promise.all([
              writeFile(`${file}.br`, br),
              writeFile(`${file}.gz`, gz),
            ]);
          }),
      );
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [contentSecurityPolicy(), precompress()],
  server: {
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        configure(proxy) {
          const origin = new URL(API_TARGET).origin;
          proxy.on("proxyReq", (outgoing, incoming) => {
            // 手元の Vite の開発サーバから来た要求だけ、Origin をプロキシ先に書き換える（サーバは同じ origin からの要求しか受けない）。
            const host = incoming.headers.host;
            if (
              host &&
              LOCAL_HOST.test(host) &&
              incoming.headers.origin === `http://${host}`
            )
              outgoing.setHeader("Origin", origin);
          });
        },
      },
    },
  },
  build: {
    target: "es2022",
    rollupOptions: { output: { manualChunks: { three: ["three"] } } },
  },
});
