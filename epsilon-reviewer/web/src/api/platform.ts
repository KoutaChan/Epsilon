import { bridgeTransport } from "./bridgeTransport.ts";
import { createApi } from "./client.ts";
import { httpTransport } from "./httpTransport.ts";

/** 画面全体で使う API。送り方は起動時に 1 回だけ選び、preload のブリッジがあればデスクトップのブリッジ、無ければ HTTP で送る。 */
export const api = createApi(
  window.epsilonReviewer
    ? bridgeTransport(window.epsilonReviewer)
    : httpTransport(),
);
