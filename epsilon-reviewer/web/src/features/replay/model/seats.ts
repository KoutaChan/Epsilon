// 卓の席の数（4 人か 3 人）と、画面での席の位置。

export const SEAT_SLOTS = ["south", "east", "north", "west"] as const;
export type SeatSlot = (typeof SEAT_SLOTS)[number];

const FOUR_SEATS = [0, 1, 2, 3] as const;
const THREE_SEATS = [0, 1, null, 2] as const;

/**
 * 画面の位置（見ている席から反時計回りに自家・下家・対面・上家）ごとに、その位置の席が見ている席からいくつ目かを返す。
 * 三人打ちでは対面の位置を空ける（null にする）。
 */
export function displaySeats(seats: number): readonly (number | null)[] {
  return seats === 3 ? THREE_SEATS : FOUR_SEATS;
}

/** 基準の席から offset 席進んだ席番号。 */
export function seatAt(start: number, offset: number, seats: number): number {
  return (start + offset) % seats;
}

/** 起家または見ている席から数えた順番。 */
export function seatOffset(start: number, seat: number, seats: number): number {
  return (seat - start + seats) % seats;
}
