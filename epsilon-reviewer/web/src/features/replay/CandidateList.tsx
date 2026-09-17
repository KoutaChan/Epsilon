import type { Candidate } from "../../api/types";
import { Tile } from "../../components/Tile";
import { actionMessageKeys } from "../../language/actionMessages";
import { useLanguage } from "../../language/LanguageProvider";

export function CandidateList({
  candidates,
  portrait = false,
}: {
  candidates: readonly Candidate[];
  portrait?: boolean;
}) {
  const { t } = useLanguage();
  return (
    <>
      <div className="em-action-heading">
        <span>{t("candidate.heading")}</span>
        <small>
          {t("candidate.top", { count: Math.min(3, candidates.length) })}
        </small>
      </div>
      <ol className={portrait ? "em-review-list" : "em-action-list"}>
        {candidates.slice(0, 3).map((candidate, index) => {
          const action = t(
            candidate.tsumogiri && candidate.type === "DAHAI"
              ? "action.dahaiTsumogiri"
              : candidate.tsumogiri && candidate.type === "RIICHI_DAHAI"
                ? "action.riichiTsumogiri"
                : actionMessageKeys[candidate.type],
          );
          const rate = (candidate.probability * 100).toFixed(1);
          return (
            <li
              key={candidate.actionId}
              className={`${portrait ? "em-review-row em-review-action" : "em-action-row"} ${index === 0 ? "is-top" : ""}`}
              aria-label={`${t("candidate.description", { action, rate })}${candidate.chosen ? `, ${t("candidate.chosen")}` : ""}`}
            >
              {portrait && <span className="em-review-rank">{index + 1}</span>}
              <div className="em-review-copy">
                <div className="em-choice-line">
                  <span className="em-choice-name">{action}</span>
                  {!portrait && (
                    <span className="em-choice-rate">
                      {rate}
                      <small>%</small>
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
                <span className="em-review-value">
                  {rate}
                  <small>%</small>
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </>
  );
}
