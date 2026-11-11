package com.epsilon.prism.run;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.LedgerReport;
import java.util.List;
import java.util.Map;

/**
 * 反復ごとの、クラス別の詳しい表。reports/noise.jsonl の 1 行になり、反復の番号は報告を書く側が先頭に足す。
 *
 * @param collect クラスごとの収集の数
 * @param ledger クラスの名前ごとの帳簿の当てはめ（当てはめなければ空）
 * @param maxAbsZ 帳簿の列の z の絶対値の最大値（当てはめなければ無い）
 * @param biased z の絶対値が偏りの警告のしきい値を越えた列の名前（当てはめなければ空）
 */
public record NoiseLine(
    Map<DecisionClass, CollectReport.ClassCounts> collect,
    Map<String, LedgerReport.ClassSummary> ledger,
    @Key(doc = "帳簿の列の z の絶対値の最大値", presence = Presence.ABSENT) Double maxAbsZ,
    List<String> biased) {

  /** reports に置くファイルの名前。 */
  public static final String FILE = "noise.jsonl";
}
