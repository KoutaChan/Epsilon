import { useMemo, type MouseEventHandler } from "react";
import { ArrowLeft, MoreHorizontal, Share2 } from "lucide-react";
import { playerName } from "../../language/format";
import { LanguageSelect } from "../../language/LanguageSelect";
import { useLanguage } from "../../language/LanguageProvider";
import { summarizePlayerDecisions } from "./model/decisionStats.ts";
import type { TableElements } from "./tableElements.ts";
import type { Playback } from "./usePlayback.ts";

/** 再生の画面の設定メニュー。見ている席の解析の指標、全員の手牌の表示、再生の位置、言語、モデル、共有のボタン、戻るボタンを並べる。 */
export function OptionsMenu({
  elements,
  playback,
  open,
  onToggle,
  onClose,
  onBack,
  backLabel,
  onShare,
}: {
  elements: TableElements;
  playback: Playback;
  open: boolean;
  onToggle: MouseEventHandler<HTMLElement>;
  onClose: () => void;
  onBack: () => void;
  backLabel: string;
  onShare: (() => void) | null;
}) {
  const { t, locale } = useLanguage();
  const { model, snapshot, seat, stepIndex, last, reveal, dispatch } = playback;
  const statistics = useMemo(() => summarizePlayerDecisions(model), [model]);
  const formats = useMemo(
    () => ({
      percent: new Intl.NumberFormat(locale, {
        style: "percent",
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }),
      loss: new Intl.NumberFormat(locale, {
        minimumFractionDigits: 3,
        maximumFractionDigits: 3,
      }),
    }),
    [locale],
  );
  const playerStats = statistics[seat]!;
  const name = playerName(snapshot.players[seat]!.name, seat, t);
  const formatDecisionRate = (count: number) =>
    playerStats.reviewed === 0
      ? t("replay.noValue")
      : formats.percent.format(count / playerStats.reviewed);
  const noDecisions =
    playerStats.reviewed === 0 ? t("replay.noDecisions") : undefined;
  // 価値を推定したモデルでは判断を損失で数えるので、説明の文も切り替える。
  const valued = playerStats.meanLoss !== null;
  return (
    <details className="em-info" open={open}>
      <summary
        ref={elements.control}
        onClick={onToggle}
        aria-label={t("replay.options")}
      >
        <MoreHorizontal />
      </summary>
      <div className="em-info-content">
        <section
          className="em-decision-stats"
          aria-label={t("replay.statistics")}
        >
          <strong className="em-stats-player" title={name}>
            {name}
          </strong>
          <span className="em-stats-scope">
            {t("replay.statisticsScope", {
              count: playerStats.reviewed.toLocaleString(locale),
            })}
          </span>
          <dl>
            <div
              title={t(
                valued
                  ? "replay.valueAgreementDescription"
                  : "replay.agreementDescription",
              )}
            >
              <dt>{t("replay.agreementRate")}</dt>
              <dd aria-label={noDecisions}>
                {formatDecisionRate(playerStats.matched)}
              </dd>
            </div>
            <div
              title={t(
                valued
                  ? "replay.valueBadMoveDescription"
                  : "replay.badMoveDescription",
              )}
            >
              <dt>{t("replay.badMoveRate")}</dt>
              <dd aria-label={noDecisions}>
                {formatDecisionRate(playerStats.badMoves)}
              </dd>
            </div>
            {playerStats.meanLoss !== null && (
              <div title={t("replay.meanLossDescription")}>
                <dt>{t("replay.meanLoss")}</dt>
                <dd>{formats.loss.format(playerStats.meanLoss)}</dd>
              </div>
            )}
          </dl>
        </section>
        <label className="em-reveal-option">
          <span>{t("replay.reveal")}</span>
          <input
            className="em-reveal-hands"
            type="checkbox"
            checked={reveal}
            onChange={(event) =>
              dispatch({ type: "reveal", reveal: event.target.checked })
            }
          />
        </label>
        <span className="em-position">
          {t("replay.positionValue", {
            current: stepIndex + 1,
            total: last + 1,
          })}
        </span>
        <label className="em-seek-label" htmlFor="review-seek">
          {t("replay.position")}
        </label>
        <input
          id="review-seek"
          className="em-timeline"
          type="range"
          min="0"
          max={last}
          step="1"
          value={stepIndex}
          onChange={(event) =>
            dispatch({ type: "seek", index: Number(event.target.value) })
          }
        />
        <LanguageSelect />
        <p>
          {t("replay.modelName", {
            name: model.model.displayName,
            version: model.model.version,
          })}
        </p>
        <div className="em-review-actions">
          {onShare && (
            <button
              className="review-home-button"
              onClick={() => {
                onClose();
                onShare();
              }}
            >
              <Share2 />
              {t("share.title")}
            </button>
          )}
          <button className="review-home-button" onClick={onBack}>
            <ArrowLeft />
            {backLabel}
          </button>
        </div>
        <p className="tile-credit">{t("replay.credit")}</p>
      </div>
    </details>
  );
}
