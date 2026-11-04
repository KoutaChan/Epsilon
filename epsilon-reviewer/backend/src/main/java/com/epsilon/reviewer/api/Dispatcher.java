package com.epsilon.reviewer.api;

import com.epsilon.reviewer.Reviewer;
import com.epsilon.reviewer.analysis.JobId;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.ModelKey;
import com.epsilon.reviewer.record.RecordId;
import com.epsilon.reviewer.store.OwnerId;
import com.epsilon.reviewer.store.ResultId;
import java.net.URI;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;

/** 操作と要求の record から Reviewer を呼ぶ（Command の網羅の switch）。HTTP と stdio が同じ振り分けを使う。 */
public final class Dispatcher {
  private final Reviewer reviewer;
  private final URI shareOrigin;

  /** @param shareOrigin 共有リンクの主（scheme://authority）。stdio は null（共有の操作は道に無い） */
  public Dispatcher(Reviewer reviewer, URI shareOrigin) {
    this.reviewer = reviewer;
    this.shareOrigin = shareOrigin;
  }

  /**
   * @param request command.requestType() の要求
   * @param body importBytes の牌譜。他は null
   * @return JSON の応答の record。文書と添付は StoredDocument、本文の無い操作は null
   */
  public Object dispatch(Command command, Caller caller, Record request, byte[] body) {
    return switch (command) {
      case MODELS -> reviewer.models();
      case HISTORY -> reviewer.history(owner(caller));
      case IMPORT_BYTES -> reviewer.importBytes(owner(caller), body);
      case IMPORT_FILE ->
          reviewer.importFile(owner(caller), path(((FileImportRequest) request).path()));
      case IMPORT_TENHOU ->
          reviewer.importTenhou(owner(caller), ((TenhouImportRequest) request).url());
      case IMPORT_STORED -> reviewer.importStored(owner(caller), result(request));
      case START_ANALYSIS -> {
        var analysis = (AnalysisRequest) request;
        yield reviewer.startAnalysis(
            owner(caller),
            RecordId.parse(analysis.recordId()),
            new ModelKey(analysis.modelId(), analysis.revision()));
      }
      case JOB -> reviewer.job(owner(caller), JobId.parse(((JobRequest) request).jobId()));
      case CANCEL_JOB ->
          reviewer.cancelJob(owner(caller), JobId.parse(((JobRequest) request).jobId()));
      case RESULT, DOWNLOAD_RESULT -> reviewer.result(owner(caller), result(request));
      case DELETE_RESULT -> {
        reviewer.deleteResult(owner(caller), result(request));
        yield null;
      }
      case EXPORT_RESULT -> {
        var export = (ExportRequest) request;
        reviewer.exportResult(
            owner(caller),
            ResultId.parse(export.resultId(), ErrorCode.RESULT_NOT_FOUND),
            path(export.path()));
        yield null;
      }
      case SHARE_LINK -> {
        ResultId shared = reviewer.shareable(owner(caller), result(request));
        yield new ShareLink(shareOrigin + "/?share=" + shared.value());
      }
      case SHARED_RESULT ->
          reviewer.sharedResult(
              ResultId.parse(((ResultRequest) request).resultId(), ErrorCode.SHARE_NOT_FOUND));
      case SCENARIO -> {
        var scenario = (ScenarioRequest) request;
        yield reviewer.scenario(
            owner(caller),
            ResultId.parse(scenario.resultId(), ErrorCode.RESULT_NOT_FOUND),
            scenario.query());
      }
      case SHARED_SCENARIO -> {
        var scenario = (ScenarioRequest) request;
        yield reviewer.sharedScenario(
            ResultId.parse(scenario.resultId(), ErrorCode.SHARE_NOT_FOUND), scenario.query());
      }
    };
  }

  private static OwnerId owner(Caller caller) {
    if (caller instanceof Caller.Owner owner) return owner.owner();
    throw new IllegalStateException("The command needs the result owner.");
  }

  private static ResultId result(Record request) {
    return ResultId.parse(((ResultRequest) request).resultId(), ErrorCode.RESULT_NOT_FOUND);
  }

  private static Path path(String text) {
    try {
      return Path.of(text);
    } catch (InvalidPathException failure) {
      throw new ReviewerException(ErrorCode.INVALID_REQUEST, "path is not a valid path.");
    }
  }
}
