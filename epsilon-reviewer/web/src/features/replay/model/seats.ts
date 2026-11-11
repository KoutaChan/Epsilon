// 卓の席の数（4 人か 3 人）と、画面での席の位置。

const FOUR_SEATS = [0, 1, 2, 3] as const;
const THREE_SEATS = [0, 1, null, 2] as const;

/**
 * 画面の位置（見ている席から反時計回りに自家・下家・対面・上家）ごとに、その位置の席が見ている席からいくつ目かを返す。
 * 三人打ちでは対面の位置を空ける（null にする）。
 */
export function displaySeats(seats: number): readonly (number | null)[] {
  return seats === 3 ? THREE_SEATS : FOUR_SEATS;
}
