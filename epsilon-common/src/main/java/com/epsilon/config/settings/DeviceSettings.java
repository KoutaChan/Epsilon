package com.epsilon.config.settings;

/**
 * 学習・推論・GRPの各処理に割り当てるデバイスと、その検証方法を指定する。
 *
 * @param validateIndices 指定デバイスインデックスが実在することを起動時に検証するなら {@code true}
 * @param learner Decision 学習処理へ割り当てるデバイス指定。空文字列なら自動選択
 * @param inference Decision 推論へ割り当てるデバイス指定。空文字列なら自動選択
 * @param grp GRPへ割り当てるデバイス指定。空文字列なら自動選択
 */
@SettingsPrefix("epsilon.devices")
public record DeviceSettings(
    @Setting("validateIndices") @Default("false") boolean validateIndices,
    @Setting("learner") @Default("") String learner,
    @Setting("inference") @Default("") String inference,
    @Setting("grp") @Default("") String grp) {}
