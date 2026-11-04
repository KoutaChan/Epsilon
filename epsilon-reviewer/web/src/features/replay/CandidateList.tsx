import type { KeyboardEvent } from "react";
import { ChevronRight } from "lucide-react";
import { Tile } from "../../components/Tile";
import { actionMessageKeys } from "../../language/actionMessages";
import { useLanguage } from "../../language/LanguageProvider";
import { listedCandidates } from "./model/candidates.ts";
import { gradeLoss, hasValueLoss } from "./model/decisionGrade.ts";
import type { Candidate } from "./model/types.ts";
import { compareCandidates, isCall } from "./scenario/scenario.ts";

/**
 * comparison は、もしもの候補を出すときの元の局面の候補。onCall があればチー・ポンの行を押して鳴いた後の打牌へ進め、
 * afterCall はその鳴き（見出しに出して戻れるようにする）。
 */
export function CandidateList({
  candidates,
  comparison = null,
  portrait = false,
  onCall = null,
  afterCall = null,
  onLeave,
}: {
  candidates: readonly Candidate[];
  comparison?: readonly Candidate[] | null;
  portrait?: boolean;
  onCall?: ((candidate: Candidate) => void) | null;
  afterCall?: Candidate | null;
  onLeave?: () => void;
}) {
  const { t } = useLanguage();
  const valued = hasValueLoss(candidates);
  const compared = comparison && compareCandidates(comparison, candidates);
  const { top, outside } = listedCandidates(candidates, compared !== null);
  const rows = outside ? [...top, outside] : top;
  const actionName = (candidate: Candidate) =>
    t(
      candidate.tsumogiri && candidate.type === "DAHAI"
        ? "action.dahaiTsumogiri"
        : candidate.tsumogiri && candidate.type === "RIICHI_DAHAI"
          ? "action.riichiTsumogiri"
          : actionMessageKeys[candidate.type],
    );
  const lossText = (loss: number) =>
    gradeLoss(loss) === "best"
      ? t("grade.best")
      : t("candidate.loss", { loss: loss.toFixed(3) });
  return (
    <>
      <div className="em-action-heading">
        <span className="em-action-title">
          {t(valued ? "candidate.valueHeading" : "candidate.heading")}
          {(compared || afterCall) && (
            <span className="em-scenario-chip">
              {afterCall
                ? t("scenario.afterCall", { action: actionName(afterCall) })
                : t("scenario.chip")}
            </span>
          )}
        </span>
        {afterCall && onLeave ? (
          <button className="em-scenario-back" onClick={onLeave}>
            {t("common.back")}
          </button>
        ) : (
          <small>{t("candidate.top", { count: top.length })}</small>
        )}
      </div>
      <ol className={portrait ? "em-review-list" : "em-action-list"}>
        {rows.map((candidate) => {
          const rank = candidates.indexOf(candidate) + 1;
          const action = actionName(candidate);
          const rate = (candidate.probability * 100).toFixed(1);
          // 価値を推定したモデルは損失を主に示し、確率は推奨度として添える。
          const loss = candidate.valueLoss ?? 0;
          const grade = valued ? gradeLoss(loss) : null;
          const description = grade
            ? t("candidate.valueDescription", {
                action,
                grade: t(`grade.${grade}`),
                loss: loss.toFixed(3),
                rate,
              })
            : t("candidate.description", { action, rate });
          const figure = grade ? (
            <>
              {lossText(loss)}
              <small className="em-choice-preference">
                {t("candidate.preference", { rate })}
              </small>
            </>
          ) : (
            <>
              {rate}
              <small>{t("candidate.rateUnit")}</small>
            </>
          );
          const gradeClass = grade ? ` em-grade-${grade}` : "";
          const chosen = candidate.chosen && (
            <span className="em-choice-tail">{t("candidate.chosen")}</span>
          );
          // 横の一覧は牌があれば実際の選択を牌と損失の下の段に分けて候補欄の幅を抑え、牌のない見送りや和了は牌の位置に置く。
          const below = !portrait && candidate.tiles.length > 0;
          // 損失は局面ごとの最善との差なので、元の値は添えるだけにして、順位の動きで比べる。
          const entry = compared && (compared.get(candidate) ?? null);
          const move = !entry
            ? null
            : entry.rank > rank
              ? "up"
              : entry.rank < rank
                ? "down"
                : "same";
          const previousRate = entry
            ? (entry.previous.probability * 100).toFixed(1)
            : "";
          const comparisonLine = compared && (
            <div className="em-scenario-compare">
              <span>
                {!entry
                  ? t("scenario.rankNew")
                  : valued
                    ? t("scenario.compare", {
                        value: lossText(entry.previous.valueLoss ?? 0),
                        rate: previousRate,
                      })
                    : t("scenario.compareRate", { rate: previousRate })}
              </span>
              {entry && move && (
                <span data-move={move}>
                  {move === "same"
                    ? t("scenario.rankSame")
                    : t(
                        move === "up" ? "scenario.rankUp" : "scenario.rankDown",
                        {
                          from: entry.rank,
                          to: rank,
                        },
                      )}
                </span>
              )}
            </div>
          );
          // もしもの間は、チー・ポンの行を押すと鳴いた後の打牌を見られる。
          const call = isCall(candidate) ? onCall : null;
          const next = call ? t("scenario.callNext", { action }) : undefined;
          const label = candidate.chosen
            ? t("candidate.withChosen", {
                description,
                chosen: t("candidate.chosen"),
              })
            : description;
          return (
            <li
              key={candidate.actionId}
              className={`${portrait ? "em-review-row em-review-action" : "em-action-row"}${rank === 1 ? " is-top" : ""}${candidate === outside ? " is-outside" : ""}${call ? " is-callable" : ""}`}
              aria-label={
                next ? t("candidate.withNext", { label, next }) : label
              }
              {...(call && {
                role: "button",
                tabIndex: 0,
                title: next,
                onClick: () => call(candidate),
                onKeyDown: (event: KeyboardEvent) => {
                  if (event.key !== "Enter" && event.key !== " ") return;
                  event.preventDefault();
                  call(candidate);
                },
              })}
            >
              {portrait && <span className="em-review-rank">{rank}</span>}
              <div className="em-review-copy">
                <div className="em-choice-line">
                  <span className="em-choice-name">
                    {action}
                    {call && (
                      <ChevronRight
                        className="em-scenario-next"
                        aria-hidden="true"
                      />
                    )}
                  </span>
                  {!portrait && (
                    <span className={`em-choice-rate${gradeClass}`}>
                      {figure}
                    </span>
                  )}
                </div>
                {(candidate.tiles.length > 0 || chosen) && (
                  <div className="em-choice-tiles">
                    {candidate.tiles.map((tile, tileIndex) => (
                      <Tile tile={tile} key={tileIndex} />
                    ))}
                    {!below && chosen}
                  </div>
                )}
                {below && chosen}
                {comparisonLine}
              </div>
              {portrait && (
                <span className={`em-review-value${gradeClass}`}>{figure}</span>
              )}
            </li>
          );
        })}
      </ol>
    </>
  );
}
