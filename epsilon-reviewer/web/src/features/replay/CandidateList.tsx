import type { Candidate } from "../../api/types";
import { Tile } from "../../components/Tile";
import { actionMessageKeys } from "../../language/actionMessages";
import { useLanguage } from "../../language/LanguageProvider";
import { gradeLoss, hasValueLoss } from "./decisionGrade";

export function CandidateList({
  candidates,
  portrait = false,
}: {
  candidates: readonly Candidate[];
  portrait?: boolean;
}) {
  const { t } = useLanguage();
  const valued = hasValueLoss(candidates);
  const top = candidates.slice(0, 3);
  // 実際の選択が上位にない場合は、区切って本来の順位で末尾に加える。
  const outside = candidates.slice(3).find((candidate) => candidate.chosen);
  const rows = outside ? [...top, outside] : top;
  return (
    <>
      <div className="em-action-heading">
        <span>
          {t(valued ? "candidate.valueHeading" : "candidate.heading")}
        </span>
        <small>{t("candidate.top", { count: top.length })}</small>
      </div>
      <ol className={portrait ? "em-review-list" : "em-action-list"}>
        {rows.map((candidate) => {
          const rank = candidates.indexOf(candidate) + 1;
          const action = t(
            candidate.tsumogiri && candidate.type === "DAHAI"
              ? "action.dahaiTsumogiri"
              : candidate.tsumogiri && candidate.type === "RIICHI_DAHAI"
                ? "action.riichiTsumogiri"
                : actionMessageKeys[candidate.type],
          );
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
              {grade === "best" ? t("grade.best") : `−${loss.toFixed(3)}`}
              <small className="em-choice-preference">
                {t("candidate.preference", { rate })}
              </small>
            </>
          ) : (
            <>
              {rate}
              <small>%</small>
            </>
          );
          const gradeClass = grade ? ` em-grade-${grade}` : "";
          return (
            <li
              key={candidate.actionId}
              className={`${portrait ? "em-review-row em-review-action" : "em-action-row"}${rank === 1 ? " is-top" : ""}${candidate === outside ? " is-outside" : ""}`}
              aria-label={`${description}${candidate.chosen ? `, ${t("candidate.chosen")}` : ""}`}
            >
              {portrait && <span className="em-review-rank">{rank}</span>}
              <div className="em-review-copy">
                <div className="em-choice-line">
                  <span className="em-choice-name">{action}</span>
                  {!portrait && (
                    <span className={`em-choice-rate${gradeClass}`}>
                      {figure}
                    </span>
                  )}
                </div>
                {(candidate.tiles.length > 0 || candidate.chosen) && (
                  <div className="em-choice-tiles">
                    {candidate.tiles.map((tile, tileIndex) => (
                      <Tile tile={tile} key={tileIndex} />
                    ))}
                    {candidate.chosen && (
                      <span className="em-choice-tail">
                        {t("candidate.chosen")}
                      </span>
                    )}
                  </div>
                )}
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
