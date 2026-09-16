# LOCO (Leave-One-Center-Out)

`fold` = そのセンターを val にする番号。fold0=center_1 ... fold5=center_7。
目的: テストに含まれるとみられる **未知センター (center_5)** での性能低下を見積もる。
v2/folds.csv と同じ 526 行・同じ列。case_id 単位のグループ構造も維持される
(1 case は 1 center にしか属さないため、center 分割は自動的に case 分割でもある)。
