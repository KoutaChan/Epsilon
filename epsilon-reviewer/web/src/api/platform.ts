import { bridgeTransport } from "./bridgeTransport.ts";
import { createApi } from "./client.ts";
import { httpTransport } from "./httpTransport.ts";

/** 画面全体の API。輸送は起動時に 1 回だけ選ぶ（preload の橋渡しがあればデスクトップ）。 */
export const api = createApi(
  window.epsilonReviewer
    ? bridgeTransport(window.epsilonReviewer)
    : httpTransport(),
);
