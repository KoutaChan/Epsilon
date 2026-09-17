const { CURRENT_RESULT_FORMAT_VERSION } = require("../dist/result-store.js");

function resultDocument(id) {
  return {
    formatVersion: CURRENT_RESULT_FORMAT_VERSION,
    resultId: id,
    createdAt: "2026-09-15T00:00:00Z",
    metadata: {
      source: "mjai",
      fileName: "game.jsonl",
      // Preserve Unicode player names through storage and recovery.
      names: ["\u6771", "\u5357", "\u897f", "\u5317"],
      finalScores: [32000, 27000, 22000, 19000],
      roundCount: 1,
    },
    model: {
      modelId: "nano",
      revision: "fixture-v1",
      displayName: "Nano",
      version: "1",
      series: "nano",
    },
    rounds: [
      {
        id: "east-1",
        roundIndex: 0,
        honba: 0,
        dealer: 0,
        states: [
          {
            stateId: 1,
            delta: null,
            checkpoint: {
              players: Array.from({ length: 4 }, () => ({
                score: 25000,
                riichi: false,
                hand: [],
                draw: null,
                river: [],
                melds: [],
              })),
              dora: ["1m"],
              honba: 0,
              kyotaku: 0,
              remaining: 70,
              activeSeat: 0,
            },
          },
        ],
        steps: [
          {
            index: 0,
            eventIndex: 1,
            stateId: 1,
            eventType: "start_kyoku",
            actor: -1,
            decision: false,
            causeEventIndex: null,
            candidates: [],
            outcome: null,
          },
        ],
      },
    ],
  };
}

module.exports = { resultDocument };
