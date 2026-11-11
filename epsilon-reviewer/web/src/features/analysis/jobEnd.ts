import type { JobStatus } from "../../../../protocol/model.ts";
import { ApiError } from "../../api/errors.ts";

/**
 * 解析のジョブが、結果を出さずに終わったか。失敗か取り消しで終わったときと、ジョブがもう無いとき（job_not_found）に
 * true を返す。ジョブが無ければ、進み具合を読み直しても取り消しても同じ失敗になる。
 *
 * @param status 最後に読めたジョブの状態
 * @param error 進み具合を読めなかったときの失敗。読めていれば null
 */
export function jobEnded(status: JobStatus, error: unknown): boolean {
  return (
    status === "failed" ||
    status === "cancelled" ||
    (error instanceof ApiError && error.code === "job_not_found")
  );
}
