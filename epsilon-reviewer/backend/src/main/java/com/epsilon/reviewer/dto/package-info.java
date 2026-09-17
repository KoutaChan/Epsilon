/**
 * 表示・保存へ所有権を渡す値。配列や List は作成側が完成させてから渡し、その後は変更しない。 GameState の借用値だけは同期コールバック内で新しい表示値へ写し、元の状態を保持しない。
 */
package com.epsilon.reviewer.dto;
