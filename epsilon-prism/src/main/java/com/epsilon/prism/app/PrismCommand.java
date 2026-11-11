package com.epsilon.prism.app;

import com.epsilon.agent.app.Command;
import com.epsilon.prism.settings.PrismSettings;

/** PRISM のコマンド。使い方はこの順に表示する。 */
public sealed interface PrismCommand extends Command<PrismSettings>
    permits Train,
        JobsList,
        JobsRun,
        JobsRetry,
        JobsCreate,
        Report,
        Evaluate,
        DuelSigma,
        FitCohorts,
        Records,
        RefitGrp,
        SwitchGrp,
        GrpTrust,
        GrpSigma,
        PlayTenhou,
        PlayRiichi {}
