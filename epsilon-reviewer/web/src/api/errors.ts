export class ApiError extends Error {
  code: string;
  status: number;

  constructor(message: string, code = "operation_failed", status = 0) {
    super(message);
    this.code = code;
    this.status = status;
  }
}
