# vectorPy 重構 TODO

整體方向：先建立安全網，再逐步從「19 個串連 pass」遷移到「evidence + candidate + score + audit」架構。
**核心原則：每一步都要有 regression 安全網、有 revert 能力、有可量化的 before/after。**

---

## 給接手的 Claude（新 session 必讀）

這份 todo.md 是 2026-05-11 的設計討論結論。如果你是新 session 的 Claude（或人類換電腦 git pull 後接手），請**先讀完本節再開工**，否則你會錯失關鍵決策邏輯。

### 環境

- **平台**：Windows + PowerShell（也可用 Bash 工具）
- **進入點**：[vectorize.py](vectorize.py)（pipeline 主體）+ [api.py](api.py)（FastAPI wrapper，本次重構不動）
- **回歸測試圖**：[srcImg/source.png](srcImg/source.png)（1200×895，常數調校基準）、[srcImg/sg2.png](srcImg/sg2.png)（2048×2048）、[srcImg/Gemini_Generated_Image_p4kt8zp4kt8zp4kt.png](srcImg/Gemini_Generated_Image_p4kt8zp4kt8zp4kt.png)（2586×1664，含文字標註）— 已加進 git
- **驗證原則**：preview PNG 是真理，segment 數一致 ≠ 沒退化（必須目視）

### 痛點現況（為什麼要重構）

[vectorize.py](vectorize.py) 是一條 19+ 個 pass 串連的純幾何 pipeline，從 [vectorize.py:2538-2654](vectorize.py#L2538-L2654) 可以看到全貌。痛點有實證：

1. **規則互相打架**：`merge_collinear` / `manhattan_ultimate_merge` / `snap_endpoints` / `force_close_free_l_corners` / `grid_snap_endpoints` 都被**重複呼叫 2–4 次**。重複呼叫同一函式來「再收一次」是規則彼此干擾的指標 — 上一個 pass 創造的問題要靠下一個 pass 救回來。
2. **「修 A 壞 B」化石**：[vectorize.py:1191-1193](vectorize.py#L1191-L1193) 註解說明「為了修厚牆 bug，又得補一條規則保住 L 角」；[vectorize.py:1395-1400](vectorize.py#L1395-L1400) 說明「為了修 L-closure，又得加 body-anchor 例外」。
3. **mask-gated 規則越加越多**：`mask_gated_l_extend` / `extend_trunk_to_loose` / `insert_missing_connectors` / `t_snap_with_extension` 都帶 `min_support` 重新查原始 mask — 表示後段純幾何規則信用破產，必須回頭問 mask 才安全。
4. **牆有時多有時少**：根因是 [vectorize.py:160-174](vectorize.py#L160-L174) 的 `v < 80 + Otsu` 對輸入色彩分佈不穩定（參數：`v < 80`、`s < 40`、`otsu_thr <= 160`、`dark_frac >= 0.02`），且 `cv2.bitwise_and(wall, ~chromatic)` 會把窗門邊的牆抗鋸齒邊緣啃掉。

### Ablation 量化證據（[output/ablation_report.csv](output/ablation_report.csv)）

跑了 27 個 pass × 3 張圖的 ablation（[ablation.py](ablation.py)），結果：

- **7 個 pass 完全 no-op**（IOU=1.000 / dN=0 / dFree=0 across all 3 images）：
  - `snap_endpoints` 第二輪（[vectorize.py:2582](vectorize.py#L2582)）
  - `manhattan_intersection_snap` 第二次呼叫（[vectorize.py:2589](vectorize.py#L2589)）
  - `grid_snap_endpoints` 第一輪（[vectorize.py:2597](vectorize.py#L2597)）
  - `force_l_corner_closure`（[vectorize.py:2598](vectorize.py#L2598)）
  - `force_close_free_l_corners` 第一輪（[vectorize.py:2605](vectorize.py#L2605)）
  - `force_close_free_l_corners` 第二輪（[vectorize.py:2611](vectorize.py#L2611)）
  - `final_polish_short_tails`（[vectorize.py:2614](vectorize.py#L2614)）
- **5 個 pass 邊際存在**（IOU > 0.99，dN ≤ 1）：`extend_trunk_to_loose`、`extend_to_intersect`、`mask_gated_l_extend`、`axis_align`、`grid_snap_2`
- **8 個 pass 真主力**（IOU < 0.95 或 dFree ≥ 5）：`snap_colinear`（IOU 0.28 on Gemini！）、`merge_collin_1`、`cluster_parallel`、`t_junction_snap`、`manhattan_force_axis`、`fuse_close_endpoints`、`brute_force_ray`、`insert_connectors`

關鍵發現：**`fuse_close_endpoints`（最後一步）對 free endpoint 影響 +23 on Gemini** — 表示前面 18 個 pass 都沒能讓圖水密，必須最後靠暴力 fuse 兜底。違反 pipeline 設計直覺。

### 目標架構：四件事一起設計（缺一退化回 heuristic soup）

不是「換個包裝的 19 個 pass」。重點在四個支柱**同時成立**：

1. **Evidence**（不是 binary mask）— Step 1 的 wall/door/window 從單一 HSV threshold 改成 multi-detector + 連續 evidence map (0.0–1.0)。原因：binary 把所有不確定性都扔給後面 19 個 pass；evidence 讓不確定性傳到 score 函數由全域代價決定。
2. **Candidate**（不是 apply）— 每個 repair 是一個 proposal，不是直接編輯 state。原因：現在的 pass 是不可逆破壞性編輯，沒有「整體更好/更壞」的判斷機制。
3. **Score**（分項可 audit，不是黑箱總分）— Primary 4 terms（wall_evidence / opening_evidence / free_endpoint / invalid_crossing）+ Derived 5 terms（phantom / duplicate / junction / opening_attachment / manhattan_consistency）。每個被接受/拒絕的 candidate 必須輸出 delta + delta_terms。原因：黑箱總分等於沒有 audit 能力。
4. **Audit**（每個拒絕都是訓練資料）— 不是事後 debug 工具，是反饋環路：production output → audit 找出低分區域 → 增量標註 → 重新訓練 → 下一輪。原因：沒 audit 的話新架構會像現在一樣只能憑感覺調。

### 為什麼是這個步驟順序（風險遞增、信心遞增）

| 步驟 | 範圍 | 風險 | 收益 | 可 revert？ |
|---|---|---|---|---|
| 0. regression harness | 純新增（tests/） | 0 | 後面所有改動的安全網 | 直接刪檔 |
| 1. 刪 7 個 NO-OP pass | vectorize.py 局部刪除 | 極低（ablation 已證實 IOU=1.000） | pipeline 變短、debug 容易 | git revert |
| 2. door/window 改 CC+bbox | 中等範圍局部改 | 低（範圍小） | 新架構第一個原型，驗證可行性 | git revert |
| 3. Step 1 改 multi-evidence | 中等範圍 | 中（評估尺度可能漂移） | 解決「牆有時多有時少」根因 | git revert + 重建 baseline |
| 4. 高風險 pass candidate 化 | 大範圍重寫 | 中–高 | 真正脫離 heuristic soup | 一次改一個 pass、分多個 commit |
| 5. ranking model | 全域 | 高（需標註集） | 替換手調權重 | 模型可丟、回退人工 score |

**第零步必須先做**。沒它，後面任何改動都是裸奔 — 你不知道改動是修了某張圖還是壞了另一張。

### 換電腦 / 新 session 的開工流程

1. `git pull` — srcImg/ 的 3 張回歸圖、ablation 報告都在 git 裡
2. 讀完本節（你正在讀的這節）
3. **驗證 ablation 結果還在**：`python ablation.py` 應該能跑（環境裝好的話），看 7 個 NO-OP 是否還是 NO-OP
4. 開始**第零步**（regression harness）— 不要直接跳到第三或第四步
5. 第零步完成後，每一步開工前都先跑 regression、完成後也跑 regression

### 關鍵設計決策（前次討論結論，不要重新發明輪子）

- **Regression IOU 用 thin/normal/loose 三層**，不是單一 thickness。thickness 跟 baseline 鎖定的 stroke_width 連動（`normal = max(2, round(0.4 × stroke_width))`）。原因：固定 thickness 會在「太敏感」和「抓不到退化」之間二選一。
- **stroke_width 在 baseline 建立時鎖定**，後續 regression 不重估。原因：避免 current mask 改壞時連評估尺度也跟著漂。
- **IOU fail 用「跌幅」不用絕對值**（`baseline_iou - current_iou > threshold`）。原因：baseline 本身可能不是完美結果。
- **`--update-baseline` 故意做得有摩擦**：互動式確認 + FAIL case 要輸完整 token。原因：跑 regression 要快，改 baseline 要慢。
- **`num_invalid_crossings` 第一版不做**。理由：定義細節多（端點 ε、cross-type 處理、T junction 合法性），會卡住第零步。第二版再加。
- **ablation.py 與 regression.py 第一版各寫各的，不抽共用模組**。理由：第零步要小、快、可丟棄，refactor 風險先別帶進來。等兩邊都穩了再抽。
- **Macro-candidate 比 Simulated Annealing 對**。理由：我們不需要逃離真正的局部最優，需要的是承認某些 repair 是天然成對的（延伸+snap、closure+merge），讓它們作為原子操作評分。
- **Spatial gate 的 bucket size 跟 scale 連動**：`max(L_EXTEND_TOL, GAP_CLOSE_TOL) × scale`。固定像素值會讓不同尺寸圖效能差 10x 以上。
- **Score 的 9 個 term 分兩層**（primary 4 + derived 5），權重總數降到 4 個調，互相干擾少。

### 容易踩的雷

- 不要為了過 regression 而調 threshold — threshold 是品質下限不是試題答案
- 不要把 19 個 pass 直接改名成 19 個 candidate generator — 那是換皮，不是重構
- 不要跳過第零步直接做架構改動
- 不要一次重寫整條 pipeline — 每步都要可 revert
- regression FAIL 時：**先 revert，搞懂為什麼壞、再決定要不要改 baseline**
- 不要從白紙標註訓練集 — 用現有 pipeline output 當基底，人工只標 baseline 哪裡錯（active learning）

### 相關檔案速查

- [vectorize.py](vectorize.py) — pipeline 主體
- [api.py](api.py) — FastAPI wrapper（不動）
- [ablation.py](ablation.py) — pass-by-pass 量化框架
- [output/ablation_report.csv](output/ablation_report.csv) — ablation 結果原始 CSV
- [output/ablation_console.txt](output/ablation_console.txt) — ablation 結果可讀日誌
- [srcImg/source.png](srcImg/source.png) / [srcImg/sg2.png](srcImg/sg2.png) / [srcImg/Gemini_Generated_Image_p4kt8zp4kt8zp4kt.png](srcImg/Gemini_Generated_Image_p4kt8zp4kt8zp4kt.png) — 3 張回歸圖

---

## 第零步：Regression Harness（最小可用版）

目標：建立改動前後的安全網。**不負責判斷結果是否完美，只阻止重構過程中不知不覺變爛。**

### 0.1 目錄與測試案例
- [ ] 建立 `tests/cases/source/manifest.json`，指向 `srcImg/source.png`
- [ ] 建立 `tests/cases/sg2/manifest.json`，指向 `srcImg/sg2.png`
- [ ] 建立 `tests/cases/Gemini_Generated/manifest.json`，指向 `srcImg/Gemini_Generated_Image_p4kt8zp4kt8zp4kt.png`
- [ ] manifest 結構：`{"input_image": "../../srcImg/<name>"}`

### 0.2 regression.py 核心函式
- [ ] `normalize_lines(lines)` — 1 px rounding、端點 canonical 順序、依 (type, x1, y1, x2, y2) 排序、忽略非幾何欄位
- [ ] `hash_lines(lines)` — 對 normalized lines 算 sha1
- [ ] `estimate_stroke_width(wall_mask)` — `2.0 × median(distanceTransform[>0])`，clamp [2.0, 80.0]，fallback `max(2, round(min(w,h)/400))`
- [ ] `rasterize_lines(lines, shape, type_filter, thickness)` — 輸出 binary mask
- [ ] `compute_iou(mask_a, mask_b)`
- [ ] `compute_distance_metrics(mask_a, mask_b)` — distance transform，回傳 mean / p95（symmetric）
- [ ] `compute_graph_metrics(lines)` — num_segments / count_by_type / free_endpoints / num_short_segments(length<10) / total_length_by_type

### 0.3 thresholds（寫死在 regression.py，後續再外部化）
- [ ] `normal_iou_drop_max`：wall=0.010 / window=0.020 / door=0.020
- [ ] `loose_iou_drop_max`：wall=0.004 / window=0.008 / door=0.008
- [ ] `thin_iou_drop_warn`：wall=0.030 / window=0.050 / door=0.050
- [ ] `p95_distance_max = max(2.0, 0.30 × stroke_width)`
- [ ] `mean_distance_max = max(1.0, 0.12 × stroke_width)`
- [ ] `free_endpoints_increase_max = 5`
- [ ] `num_segments_change_ratio_max = 0.15`
- [ ] `total_length_change_ratio_max = 0.10`

### 0.4 status 合成
- [ ] FAIL：normal/loose IOU drop 超標 / distance 超標 / free_endpoints +>5 / num_segments 變動 >15% / total_length 變動 >10%
- [ ] WARNING：thin IOU drop 超標 / num_short_segments 增加 / hash 變但 IOU 全 pass
- [ ] PASS：其他

### 0.5 主流程
- [ ] 走純函式：`cv2.imread → vectorize_bgr(bgr) → result["lines"]`，不經 `run_one`
- [ ] 沒 baseline 時直接報錯，提示用 `--update-baseline`（**不自動建**）
- [ ] 每 case 跑完 →（拿 baseline stroke_width）→ rasterize thin/normal/loose × {wall, window, door} → compute IOU + distance + graph metrics → 判 status
- [ ] 寫 `tests/current/<case>/lines.json` + `masks/` + `metrics.json` + `hash.txt` + `report.json` + `diff_overlay.png`

### 0.6 baseline 結構
- [ ] `tests/baseline/<case>/lines.json`
- [ ] `tests/baseline/<case>/masks/{wall,window,door}_{thin,normal,loose}.png`
- [ ] `tests/baseline/<case>/metrics.json` 含 `regression_format_version=1`、`rasterizer_version=1`、`stroke_width`
- [ ] `tests/baseline/<case>/hash.txt`

### 0.7 --update-baseline 互動式安全閘
- [ ] 預設互動：跑完顯示 status + delta + overlay 路徑
- [ ] PASS / WARNING：`Update baseline for <case>? [y/N]:`（預設 No）
- [ ] FAIL：要求輸入完整 token `Type UPDATE <case> to update baseline anyway:`
- [ ] 首次建 baseline：`Type CREATE <case> to confirm:`
- [ ] `--yes` 旗標 bypass（log 記錄 `baseline_updated_by: manual_yes_flag` + timestamp）
- [ ] `--case <name>` 限定單一 case 更新
- [ ] `--all` 全部更新（仍要輸入 `UPDATE ALL`）

### 0.8 overlay
- [ ] `diff_overlay.png`：baseline-only（紅）/ current-only（綠）/ overlap（黑）三色合成
- [ ] 印出檔案路徑，不嘗試自動開圖（避免 CI / 遠端環境失敗）

### 0.9 文件
- [ ] `tests/README.md`：說明使用方式、目錄結構、status 三態定義、threshold 一覽

### 0.10 驗收
- [ ] `python regression.py` 沒 baseline 時報錯且 exit 1
- [ ] `python regression.py --update-baseline` 互動建立 3 個 case 的 baseline
- [ ] 不改 vectorize.py 再跑 `python regression.py` 應全部 PASS、exit 0
- [ ] 故意改 vectorize.py 一個 tol 製造小退化 → 驗證能抓到 WARNING / FAIL

---

## 第一步：刪除 ablation 確認的 NO-OP pass

依據今天 ablation 結果，下列 7 個 pass 在 3 張回歸圖上 IOU=1.000、dN=0、dFree=0：

- [ ] 確認每個 NO-OP pass 的引入 commit（`git log -p`），記錄它原本是為了哪張圖加的
- [ ] 從 `vectorize_bgr` 移除：
  - [ ] `snap_endpoints` 第二輪呼叫（[vectorize.py:2582](vectorize.py#L2582)）
  - [ ] `manhattan_intersection_snap` 第二次呼叫（[vectorize.py:2589](vectorize.py#L2589)）
  - [ ] `grid_snap_endpoints` 第一輪（[vectorize.py:2597](vectorize.py#L2597)）
  - [ ] `force_l_corner_closure`（[vectorize.py:2598](vectorize.py#L2598)）
  - [ ] `force_close_free_l_corners` 第一輪（[vectorize.py:2605](vectorize.py#L2605)）
  - [ ] `force_close_free_l_corners` 第二輪（[vectorize.py:2611](vectorize.py#L2611)）
  - [ ] `final_polish_short_tails`（[vectorize.py:2614](vectorize.py#L2614)）
- [ ] 對應的常數視情況保留還是刪除（如果整個函式不再被呼叫，函式本體保留但加註 TODO）
- [ ] 跑 `python regression.py` 驗證全 PASS
- [ ] commit：`refactor: remove 7 no-op passes confirmed by ablation`

---

## 第二步：door / window 改 CC + bbox（不走 skeleton）

現況：door/window 走完整 skeleton → branches → approxPolyDP → snap pipeline，繞遠路。
目標：CC → bbox → 兩條長邊 centerline → attach 最近相容牆。

- [ ] 在 `segment_colors` 之後對 door/window mask 各跑 `cv2.connectedComponentsWithStats`
- [ ] 過濾 area / aspect ratio（門窗都是長條矩形）
- [ ] 每個 component 取 minAreaRect，輸出兩條長邊作為 centerline segments
- [ ] 短邊（門框/窗框兩端）視需求保留為 jamb segments
- [ ] door/window 改走這個路徑，wall 仍走原 skeleton pipeline
- [ ] 跑 regression：window/door normal_iou drop 應在 threshold 內
- [ ] 視覺檢查 overlay，確認門窗形狀合理
- [ ] commit + `--update-baseline`（如果結果更好）

---

## 第三步：Step 1 改 multi-evidence

現況：`segment_colors` 用單一 HSV threshold + Otsu，輸出 binary mask。
目標：多 detector → 連續 evidence map（0.0–1.0），保留 debug layer。

### 3.1 多 detector
- [ ] D1 strong black：`v < v_strong`
- [ ] D2 low-saturation dark：`s < s_low and v < otsu_adaptive`
- [ ] D3 edge-supported dark stroke：local contrast + dark center
- [ ] D4 long-rect CC：connected component 篩 elongation
- [ ] (D5 PDF source detector — 看實際輸入是否 vector PDF 再決定要不要做)

### 3.2 component scoring
- [ ] darkness_score / elongation_score / rectilinearity_score / length_score / stroke_width_consistency
- [ ] 反向：text_likelihood / small_blob_penalty
- [ ] 每個 wall component 算總分，留下高分

### 3.3 evidence map 整合
- [ ] 多 detector 加權 → 連續 wall_evidence (0.0–1.0)
- [ ] 改 `wall_support` 計算：從 binary count 變成 evidence integral
- [ ] thresholding 給 skeleton 用（用 Otsu on evidence histogram，避免新手調常數）
- [ ] 保留 debug layer：可選 dump 每個 detector 的中間結果

### 3.4 驗收
- [ ] 跑 regression：所有 IOU 在 threshold 內
- [ ] **目視確認** Gemini 圖（文字標註多）的牆少抓 / 多抓有改善
- [ ] commit + `--update-baseline`

---

## 第四步：高風險 pass candidate 化

把現在「碰到就動」的高風險 pass 改成「propose → score delta → accept/reject」。先挑這 5 個：

- [ ] `insert_missing_connectors`（會發明牆）
- [ ] `brute_force_ray_extend`（沿軸亂掃）
- [ ] `extend_trunk_to_loose`（延伸 trunk）
- [ ] `mask_gated_l_extend`（不對稱 L 延伸）
- [ ] `t_snap_with_extension`（trunk 自動延伸）

### 4.1 score 函數雛形
- [ ] 設計 primary terms（4 個）：wall_evidence / opening_evidence / free_endpoint / invalid_crossing
- [ ] 設計 derived terms（5 個）：phantom / duplicate / junction / opening_attachment / manhattan_consistency
- [ ] 輸出分項 breakdown（不是只有 total）
- [ ] 對每個被接受 / 拒絕的 candidate，輸出 delta + delta_terms

### 4.2 spatial gate
- [ ] uniform grid bucket，bucket size = `max(L_EXTEND_TOL, GAP_CLOSE_TOL) × scale`
- [ ] H_buckets / V_buckets 各自存 axis line + interval
- [ ] 自由端點查詢只掃附近 bucket

### 4.3 semantic gate
- [ ] wall ↔ wall：可 gap close / T snap
- [ ] door/window ↔ wall trunk：可 attach
- [ ] door/window ↔ door/window：通常不產生 repair
- [ ] 列表化合法組合，不在表上的不產生 candidate

### 4.4 evidence gate
- [ ] gap connector：corridor mask_support < 0.3 不產生
- [ ] 分級：>=0.75 high / 0.45–0.75 maybe / <0.45 reject

### 4.5 改造順序
- [ ] 一次改一個 pass，每改一個跑 regression
- [ ] **每個 pass 改完都 commit**，方便 revert
- [ ] 全部改完跑 ablation，看每個 candidate-based pass 的 IOU / dN / dFree

### 4.6 macro-candidate（compound repair）
- [ ] 把常見成對 repair 包成原子：`extend_then_snap`、`close_L_with_two_extensions`、`gap_close_and_merge`
- [ ] bounded lookahead 觸發條件：A 解決 ≥50% 一個 issue 才考慮 A+B

---

## 第五步：scoring + ranking model（最後做）

前 4 步穩定後再進。

### 5.1 標註集
- [ ] **不從白紙標註**：用現有 pipeline output 當基底，人工只標 baseline 哪裡錯
- [ ] 每張 5–10 分鐘可行，30–50 張當訓練資料
- [ ] 格式：`expected_walls.json` / `expected_doors.json` / `expected_windows.json`

### 5.2 candidate 特徵
- [ ] mask_support / length / synthetic / reduces_free_endpoint / creates_crossing / attaches_opening / duplicate_overlap

### 5.3 模型
- [ ] 先 logistic regression 或 gradient boosted trees，**不要先上深度學習**
- [ ] 訓練目標：candidate accept probability
- [ ] 替換手調 αβγ：`accept if P > 0.85`

### 5.4 audit loop
- [ ] production output → audit identifies low-score region → 增量標註 → retrain → 下一輪

---

## 持續性原則

- 每步前都跑 regression、每步後都跑 regression
- regression FAIL 時：**先 revert，搞懂為什麼壞、再決定要不要改 baseline**
- baseline update 故意做得有摩擦：跑 regression 要快，改 baseline 要慢
- 不要為了過 regression 而調 threshold，threshold 是品質下限不是試題答案
- 每個 commit 都應該有 ablation / regression 數字佐證
