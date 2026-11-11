import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  RiverEdit,
  ScenarioAnswer,
  ScenarioEdits,
  ScenarioQuery,
} from "../../../../../protocol/model.ts";
import type { TableState } from "../../../../../protocol/result.ts";
import type { ScenarioClient } from "../../../review-entry.ts";
import { readCandidates } from "../model/candidates.ts";
import type { Candidate, DecisionStep, Snapshot } from "../model/types.ts";
import type { Playback } from "../usePlayback.ts";
import {
  NO_EDITS,
  applyEdits,
  canRevert,
  listChanges,
  revert,
  setDora,
  setHand,
  setRiver,
  setScore,
  setSticks,
  tableSnapshot,
  type EditedBoard,
  type ScenarioChange,
  type ScenarioTarget,
} from "./scenario.ts";

export type ScenarioBoard = "edited" | "original";
export type ScenarioStatus =
  "idle" | "ready" | "loading" | "done" | "stale" | "error";
type Analysis =
  | { status: "loading"; key: string }
  | { status: "done"; key: string; candidates: readonly Candidate[] }
  | { status: "error"; key: string; error: unknown };
/** 打牌への応答でチー・ポンしたとして、その後の打牌を解析した結果。`call` はそのチー・ポンの候補。 */
export type ScenarioBranch =
  | { call: Candidate; status: "loading" }
  | {
      call: Candidate;
      status: "done";
      candidates: readonly Candidate[];
      table: TableState | undefined;
    };
interface Draft {
  edits: ScenarioEdits;
  analysis: Analysis | null;
  branch: ScenarioBranch | null;
  /** 鳴いた後の解析が失敗した理由。候補は元に戻し、理由を右上に出す。 */
  callError: unknown;
}
const EMPTY: Draft = {
  edits: NO_EDITS,
  analysis: null,
  branch: null,
  callError: null,
};
/** 変更も鳴いた後の分岐も失敗も無い下書きは捨てる。変更が無ければ解析の結果は出さないので、残しても使わない。 */
const isEmpty = (draft: Draft) =>
  draft.edits === NO_EDITS && draft.branch === null && draft.callError === null;

export interface Scenario {
  /** 解析結果を出したモデルで、もしもを解析できるか。できなければ、もしもは押せない。 */
  readonly available: boolean;
  readonly active: boolean;
  /** もしもを始められるか。見ている席が判断する局面でだけ始められる。 */
  readonly canStart: boolean;
  toggle(): void;
  readonly decision: DecisionStep | null;
  readonly board: ScenarioBoard;
  showBoard(board: ScenarioBoard): void;
  /** 卓の上に当たり判定を出して編集できるか。元の局面や鳴いた後の局面を見ている間は編集させない。 */
  readonly editing: boolean;
  readonly selection: ScenarioTarget | null;
  select(target: ScenarioTarget | null): void;
  readonly original: Snapshot;
  readonly edited: EditedBoard;
  /** 卓に描く局面。 */
  readonly display: Snapshot;
  readonly changes: readonly ScenarioChange[];
  readonly status: ScenarioStatus;
  readonly error: unknown;
  analyse(): void;
  /** 鳴いた後の打牌を見ている間の、そのチー・ポンと解析の状態（読み込み中を含む）。見ていなければ null。 */
  readonly branch: ScenarioBranch | null;
  /** 応答の判断で、チー・ポンの候補から鳴いた後の打牌へ進む。進めないときは null。 */
  readonly branchTo: ((call: Candidate) => void) | null;
  leaveBranch(): void;
  /** 候補の欄に出す状態（読み込み中かどうかなど）。 */
  readonly listStatus: ScenarioStatus;
  /** 候補の欄に出す候補。解析が今の変更に合うときだけ、もしもの候補にする。 */
  readonly candidates: readonly Candidate[];
  /** もしもの候補を出しているときの、比べる元の候補。鳴いた後の打牌には元が無いので null。 */
  readonly comparison: readonly Candidate[] | null;
  setRiver(
    seat: number,
    index: number,
    change: Partial<Pick<RiverEdit, "tile" | "tsumogiri">>,
  ): void;
  setDora(index: number, tile: string): void;
  setHand(index: number, tile: string): void;
  setScore(seat: number, score: number): void;
  setSticks(kind: "honba" | "kyotaku", count: number): void;
  revert(target: ScenarioTarget): void;
  /** 元に戻せるか。元の牌に戻すと、見えている枚数が物理牌の数を超える所は戻せない。 */
  canRevert(target: ScenarioTarget): boolean;
  revertAll(): void;
}

// 変更と解析の結果は、判断（局・イベント・席）ごとに画面の中だけで覚えておく。
export function useScenario(
  playback: Playback,
  client: ScenarioClient | null,
): Scenario {
  const { model, roundIndex, seat, decision, snapshot, dispatch } = playback;
  const [active, setActive] = useState(false);
  const [board, setBoard] = useState<ScenarioBoard>("edited");
  const [selection, setSelection] = useState<ScenarioTarget | null>(null);
  const [drafts, setDrafts] = useState<ReadonlyMap<string, Draft>>(
    () => new Map(),
  );
  const requests = useRef(new Map<string, AbortController>());
  const key = decision && `${roundIndex}:${decision.eventIndex}:${seat}`;
  // 選んでいる所と、元の局面を見ているかは、その判断だけのもの。別の判断へ移ったら、その描画のうちに初めへ戻す
  // （前の選択のままでは何も描かない）。
  const [viewed, setViewed] = useState(key);
  if (viewed !== key) {
    setViewed(key);
    setSelection(null);
    setBoard("edited");
  }
  const draft = (key !== null && drafts.get(key)) || EMPTY;
  const { branch } = draft;
  const edited = useMemo(
    () => applyEdits(snapshot, draft.edits, seat),
    [snapshot, draft.edits, seat],
  );
  const afterCall = useMemo(
    () =>
      branch?.status === "done" && branch.table
        ? tableSnapshot(snapshot, branch.table)
        : null,
    [snapshot, branch],
  );
  const changes = useMemo(
    () => listChanges(draft.edits, snapshot, seat),
    [draft.edits, snapshot, seat],
  );
  const editsKey = JSON.stringify(draft.edits);
  useEffect(() => {
    const pending = requests.current;
    return () => {
      for (const controller of pending.values()) controller.abort();
    };
  }, []);
  const update = useCallback(
    (target: string, change: (current: Draft) => Draft) =>
      setDrafts((previous) => {
        const next = change(previous.get(target) ?? EMPTY);
        const drafts = new Map(previous);
        if (isEmpty(next)) drafts.delete(target);
        else drafts.set(target, next);
        return drafts;
      }),
    [],
  );
  // 編集の吹き出しは外のクリックを監視しているので、いつも同じ関数を渡す。
  const select = useCallback((target: ScenarioTarget | null) => {
    setSelection(target);
    if (target) setBoard("edited");
  }, []);
  const cancel = (slot: string) => {
    requests.current.get(slot)?.abort();
    requests.current.delete(slot);
  };
  // 局面を変えたら、鳴いた後の解析は前の局面に対するものなので捨てる。
  const edit = (change: (edits: ScenarioEdits) => ScenarioEdits) => {
    if (key === null) return;
    cancel(`${key}:call`);
    update(key, (current) => ({
      ...current,
      edits: change(current.edits),
      branch: null,
      callError: null,
    }));
    setBoard("edited");
  };
  /** サーバに解析を要求する。同じ判断で同じ種類（局面・鳴き）の前の要求は取り消す。 */
  const send = (
    slot: string,
    query: ScenarioQuery,
    done: (answer: ScenarioAnswer) => void,
    failed: (error: unknown) => void,
  ) => {
    if (client === null) return;
    cancel(slot);
    const controller = new AbortController();
    requests.current.set(slot, controller);
    client
      .evaluate(query, controller.signal)
      .then(
        (answer) => {
          if (!controller.signal.aborted) done(answer);
        },
        (error: unknown) => {
          if (!controller.signal.aborted) failed(error);
        },
      )
      .finally(() => {
        if (requests.current.get(slot) === controller)
          requests.current.delete(slot);
      });
  };
  const analysis = draft.analysis;
  const status: ScenarioStatus =
    analysis?.status === "loading"
      ? "loading"
      : analysis !== null && analysis.key === editsKey
        ? analysis.status
        : changes.length === 0
          ? "idle"
          : analysis?.status === "done"
            ? "stale"
            : "ready";
  const analyse = () => {
    if (key === null || decision === null || changes.length === 0) return;
    const requestKey = editsKey;
    update(key, (current) => ({
      ...current,
      analysis: { status: "loading", key: requestKey },
    }));
    send(
      `${key}:position`,
      {
        round: roundIndex,
        eventIndex: decision.eventIndex,
        seat,
        edits: draft.edits,
      },
      (answer) =>
        update(key, (current) => ({
          ...current,
          analysis: {
            status: "done",
            key: requestKey,
            candidates: readCandidates(answer, model.temperature),
          },
        })),
      (error) =>
        update(key, (current) => ({
          ...current,
          analysis: { status: "error", key: requestKey, error },
        })),
    );
  };
  /** 応答の判断で、そのチー・ポンをした後の打牌を解析して卓をその局面にする。 */
  const branchTo = (call: Candidate) => {
    if (key === null || decision === null) return;
    // 答えが来る前に戻ったり局面を変えたりしたら、その答えは捨てる。
    const settle = (next: Partial<Draft>) =>
      update(key, (current) =>
        current.branch?.call === call ? { ...current, ...next } : current,
      );
    update(key, (current) => ({
      ...current,
      branch: { call, status: "loading" },
      callError: null,
    }));
    setSelection(null);
    setBoard("edited");
    send(
      `${key}:call`,
      {
        round: roundIndex,
        eventIndex: decision.eventIndex,
        seat,
        edits: draft.edits,
        call: call.actionId,
      },
      (answer) =>
        settle({
          branch: {
            call,
            status: "done",
            candidates: readCandidates(answer, model.temperature),
            table: answer.table,
          },
        }),
      (error) => settle({ branch: null, callError: error }),
    );
  };
  const showEdited = active && board === "edited" && decision !== null;
  const branched = showEdited && branch !== null;
  const outcome =
    showEdited && status === "done" && analysis?.status === "done"
      ? analysis.candidates
      : null;
  // 今の変更に合う解析の失敗だけを出す（変えた後に古い失敗を残さない）。
  const error =
    draft.callError ??
    (status === "error" && analysis?.status === "error"
      ? analysis.error
      : null);
  const available = client !== null;
  return {
    available,
    active,
    canStart: available && decision !== null,
    toggle: () => {
      setSelection(null);
      if (active) setActive(false);
      else if (available && decision !== null) {
        dispatch({ type: "pause" });
        setActive(true);
        setBoard("edited");
      }
    },
    decision,
    board,
    showBoard: setBoard,
    editing: showEdited && !branched,
    selection,
    select,
    original: snapshot,
    edited,
    display:
      branched && afterCall
        ? afterCall
        : showEdited
          ? edited.snapshot
          : snapshot,
    changes,
    status,
    error,
    analyse,
    branch: branched ? branch : null,
    branchTo: showEdited && !branched ? branchTo : null,
    leaveBranch: () => {
      if (key === null) return;
      cancel(`${key}:call`);
      update(key, (current) => ({ ...current, branch: null }));
    },
    listStatus: branched && branch.status === "loading" ? "loading" : status,
    candidates:
      branched && branch.status === "done"
        ? branch.candidates
        : (outcome ?? playback.candidates),
    comparison: outcome && !branched ? playback.candidates : null,
    setRiver: (target, index, change) =>
      edit((edits) => setRiver(edits, snapshot, target, index, change)),
    setDora: (index, tile) =>
      edit((edits) => setDora(edits, snapshot, index, tile)),
    setHand: (index, tile) =>
      edit((edits) => setHand(edits, snapshot, seat, index, tile)),
    setScore: (target, score) =>
      edit((edits) => setScore(edits, snapshot, target, score)),
    setSticks: (kind, count) =>
      edit((edits) => setSticks(edits, snapshot, kind, count)),
    revert: (target) => edit((edits) => revert(edits, snapshot, target)),
    canRevert: (target) => canRevert(edited, snapshot, seat, target),
    revertAll: () => edit(() => NO_EDITS),
  };
}
