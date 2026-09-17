import { defineConfig } from "vite";
const contentSecurityPolicy =
  "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; form-action 'self'";
export default defineConfig(({ command }) => ({
  base: "./",
  plugins:
    command === "build"
      ? [
          {
            name: "production-csp",
            transformIndexHtml() {
              return [
                {
                  tag: "meta",
                  attrs: {
                    "http-equiv": "Content-Security-Policy",
                    content: contentSecurityPolicy,
                  },
                  injectTo: "head-prepend",
                },
              ];
            },
          },
        ]
      : [],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (outgoing, incoming) => {
            // ローカルVite自身から来たリクエストだけ、開発用の接続先へOriginを揃える。
            const host = incoming.headers.host,
              origin = incoming.headers.origin;
            if (
              host &&
              /^(localhost|127\.0\.0\.1)(:\d+)?$/.test(host) &&
              origin === `http://${host}`
            ) {
              outgoing.setHeader("Origin", "http://127.0.0.1:8080");
            }
          });
        },
      },
    },
  },
  build: {
    target: "es2022",
    rollupOptions: { output: { manualChunks: { three: ["three"] } } },
  },
}));
