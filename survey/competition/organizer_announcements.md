# Organizer Announcements — Tiger SQ-AI Challenge (原文保存)

主催者 (Max, on behalf of the Tiger SQ-AI Challenge Organizers) からのアナウンス原文。
**ここに書かれている事項は主催者が既に把握済み**なので、こちらから不具合として指摘しないこと。

---

## 最新: 訓練データ新バージョン公開（第3バッチの確定版）

> 📢 News
>
> Dear participants,
> We have uploaded a new version of the TIGER SQ-AI 2026 training dataset to the Synapse project. If you have already downloaded the data, please pull it again before your next training run.
>
> **What Changed**
> - **Broader coverage.** The training set now spans **40 cases and 524 annotated frames**, up from 28 cases. Most of the new material comes from centers 2, 3, 4 and 7.
> - **Refined segmentations.** **30 fine masks were updated** after a second review round with our clinical reviewers, **including several corrected nerve labels**. The coarse masks were regenerated accordingly.
> - **Corrected station visibility.** `lymph_node_station_visibility.csv` now contains **518 rows**. Following a renewed review of **stations 11R and 11L**, the visible station lists were revised on **nine cases**. **In every instance, stations were added, never removed.**
> - `labelmap.csv` is unchanged, so no changes are needed to your class mapping.
>
> **One Note If You Parse Filenames**
> Two training frames are published as **extra frames** rather than as a station's selected frame, **because the chosen frame did not in fact show the station it was selected for**:
> - `center_3_case_3_na_station_1.png`
> - `center_7_case_3_na_station_1.png`
>
> These carry **no row** in `lymph_node_station_visibility.csv`. Every other training frame keeps the `center_<n>_case_<m>_<station>.png` convention, and **the test input your container receives at `/input` will contain only that form**, so this affects training time parsing only.
>
> **Early Test Submissions Welcome**
> We would like to verify that submitted containers run correctly on our infrastructure well before the deadline. If your pipeline is far enough along, please feel free to send us a submission now, even a partial or placeholder one. A container that simply writes correctly formatted output is enough for us to validate the plumbing end to end. We will report back what we see and help resolve any issues.
>
> Please follow the interface described in the submission instructions (`/output/task1/`, `/output/task2/`, `/output/task3.csv`). You may enter one, two, or all three tasks.
>
> **Timeline**
> 🏁 September 15th, 2026: Submission deadline

---

## 第3バッチ公開

> Dear Tiger SQ-AI Challenge Participants,
> We are pleased to announce that the 3rd batch of data is now online.
>
> Here is the current status of the datasets:
> - **Tasks 1 and 2: The training data is now complete.**
> - **Task 3:** We have uploaded an initial set of new examples and will release the final update next week. Please note that, **depending on our clinical team's annotation progress, the final dataset for Task 3 may have fewer total examples compared to Tasks 1 and 2.** We will inform you about the progress.

---

## 第3バッチ遅延の連絡

> We wanted to let you know that the 3rd batch of data will be published later this week.
> We apologize for the slight delay and greatly appreciate your patience as we finalize the release.

---

## write-up / Docker 手順の公開・タイムライン改訂（7/17）

> We are writing to inform you that the instructions for the write-up and Docker container generation have been uploaded. If you have any questions regarding these materials, please feel free to use the Synapse page discussion forum or reach out to us via email.
>
> Additionally, we have successfully uploaded **5 new training cases** today. We sincerely apologize that we cannot provide more frames at this stage. The annotation process has proven to be much more complex and time-consuming than we initially anticipated, and we appreciate your patience.
>
> To ensure a smooth and successful challenge for everyone, we have updated the timeline as follows:
> - **July 17 (Today):** 5 additional annotated cases uploaded.
> - **around August 15:** Additional training data will be uploaded. **Docker container submission opens.**
>   *Note: We highly encourage you to submit your Docker image early to test compatibility with our setup and receive feedback. Early submissions also help us verify that our evaluation scripts are running smoothly.*
> - **September 6:** The evaluation phase begins. Again, we strongly suggest submitting your solution as early as possible to guarantee a working submission.
> - **September 15, 2026:** Final submission deadline.
>
> Please also note that **MICCAI has been relocated to Strasbourg.** Our challenge presentation is scheduled for **September 27 during the EndoVis 2026 session.**
> Top-performing teams are warmly invited to present their work at MICCAI!
> We kindly ask you to prepare a **3-minute video presentation** of your submitted method (**one combined presentation for Tasks 1 and 2, and a separate presentation for Task 3**) which must be uploaded with the Challenge write-up.

---

## ここから導かれる、こちらの前提

| 事項 | 確定内容 |
|---|---|
| 公式のフレーム数 | **524**（ローカルのファイル数 528 = 524 + `na_station_1` 2 枚 + `_frame_` 2 枚） |
| `na_station_1` 2 枚 | **仕様。** 選定フレームが当該 station を写していなかったため extra frame として公開。Task3 行が無いのも仕様 |
| Task3 の 518 行 | **現行の正。** 11R/11L の見直しで 9 case を修正、**追加のみ・削除なし**。Task1/2 より少なくなりうると明言済み |
| テスト入力のファイル名 | **`center_<n>_case_<m>_<station>.png` の形式のみ**。`na_station` や `_frame_` は来ない → 提出コンテナのパースは標準形だけ考えればよい |
| fine マスク 30 枚更新 | 神経ラベルの修正を含む。**古いバージョンで学習した重みは再学習対象** |
| labelmap.csv | 不変 |
| 早期提出 | **主催者が明確に推奨**。プレースホルダでも可、フィードバックをくれる |
