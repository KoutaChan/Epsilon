package com.epsilon.classic.app;

import com.epsilon.agent.app.Command;

/** classic のコマンド。使い方には、ここに並べた順で表示する。 */
public sealed interface ClassicCommand extends Command<ClassicSettings>
    permits Init, Rows, Pretrain, PretrainGrp, Campaign, Champion {}
