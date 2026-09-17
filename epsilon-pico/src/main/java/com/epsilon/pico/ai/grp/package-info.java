/**
 * 公開された局系列から半荘終了時の順位分布を推定する Global Rank Predictor を提供する。
 *
 * <p>Decision 学習では固定チェックポイントを局境界教師として参照し、自己対局の Decision オプティマイザーから GRP パラメーター を更新しない。 GRP
 * 自身の事前学習・検証・チェックポイントトランザクションは独立して扱う。
 */
package com.epsilon.pico.ai.grp;
