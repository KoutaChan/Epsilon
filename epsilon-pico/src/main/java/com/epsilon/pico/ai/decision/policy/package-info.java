/**
 * 動的合法候補のスコアと麻雀固有の条件付き方策グラフを提供する。
 *
 * <p>RON・TSUMO・九種九牌などの受理二択の判定、PASS 対 MELD、CONTINUE 対 KAN、種別、具体候補を別節点として正規化する。 FULL_SUPPORT
 * はこのモデル分布へ探索に割り当てる確率を混ぜるサンプリング層であり、候補効用や条件付きスコア計算処理の定義には混ぜない。
 */
package com.epsilon.pico.ai.decision.policy;
