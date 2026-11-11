package com.epsilon.agent.app;

/** 全てのアプリにある組み込みのコマンド。どのアプリの設定でも動く。 */
public sealed interface BuiltinCommand extends Command<Record>
    permits Help, SettingsShow, SettingsTemplate, SettingsCreate, SettingsChange, SettingsCheck, Status, ControlCommand {}
