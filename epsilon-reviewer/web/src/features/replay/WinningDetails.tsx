import type { WinDetails } from "../../api/types";
import { Tile } from "../../components/Tile";
import { useLanguage } from "../../language/LanguageProvider";
import { yakuLabel } from "../../language/yakuMessages";

export function WinningDetails({
  details,
  dora,
}: {
  details: WinDetails | null;
  dora: readonly string[];
}) {
  const { t, locale } = useLanguage();
  const yakumanLabel = (count: number) =>
    t(count === 1 ? "outcome.yakuman" : "outcome.multipleYakuman", { count });
  return (
    <div className="em-winning-details">
      {details && (
        <>
          <div className="em-winning-value">
            {details.points !== null && (
              <strong>
                {details.points.toLocaleString(locale)}
                <small>{t("outcome.points")}</small>
              </strong>
            )}
            <span>
              {details.yakuman
                ? yakumanLabel(details.yakuman)
                : [
                    details.han !== null
                      ? t("outcome.han", { count: details.han })
                      : "",
                    details.fu !== null
                      ? t("outcome.fu", { count: details.fu })
                      : "",
                  ]
                    .filter(Boolean)
                    .join(" · ")}
            </span>
          </div>
          {details.yaku && details.yaku.length > 0 && (
            <dl className="em-winning-yaku" aria-label={t("outcome.yaku")}>
              {details.yaku.map((yaku, index) => (
                <div
                  key={`${yaku.code}-${index}`}
                  data-bonus={yaku.code.endsWith("DORA")}
                >
                  <dt>{yakuLabel(yaku.code, t)}</dt>
                  <dd>
                    {yaku.yakuman > 0
                      ? yakumanLabel(yaku.yakuman)
                      : t("outcome.han", { count: yaku.han })}
                  </dd>
                </div>
              ))}
            </dl>
          )}
        </>
      )}
      <div className="em-winning-indicators">
        <div className="em-result-dora">
          <span>{t("outcome.doraIndicators")}</span>
          <div>
            {dora.map((tile, index) => (
              <Tile key={index} tile={tile} />
            ))}
          </div>
        </div>
        {details?.uraIndicators && details.uraIndicators.length > 0 && (
          <div className="em-result-dora em-result-ura">
            <span>{t("outcome.uraIndicators")}</span>
            <div>
              {details.uraIndicators.map((tile, index) => (
                <Tile key={index} tile={tile} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
