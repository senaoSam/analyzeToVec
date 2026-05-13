# vectorPy 下一階段 TODO

**核心方針**:從「靠 17 道串聯補洞 + bit-identical baseline 維持」轉到「靠語意 invariant + 品質 metric + 連續 score 維持」。
舊架構的根本問題:bit-identical baseline 強迫每一次改進偽裝成 NO-OP,任何「明顯變化但是優化」的修法都會被擋住、然後被分解成多個下游補丁去喬 byte。

---

## 接受條件(scope contract)

任何修改只要滿足下列**任一**就視為改進、可接受:
- 所有品質 metric 都相等或更好,且 invariant 不破
- 某個 metric 明顯變好(IOU ↑、`floating_openings` ↓、`free_endpoints` ↓),其他 metric 在容差內

**不接受**:
- 任何 invariant 被破(見 §1)
- 某個 metric 為了讓另一個 metric 變好而明顯退化,且沒辦法 user-approve

「肉眼看不見的變化」與「肉眼看得見但更貼近原圖/補上懸空」都算改進。

---

## §1 · Step 18 — Metric-based regression(基礎建設,先做)

**動機**:目前 [regression.py](regression.py) 是 bit-identical baseline,改 A 壞 B 的儀式成本來自這裡。換成 metric-based,後續所有 step 才有意義的判準。

### 18.1 — Invariant 斷言層(硬紅線,永遠不接受違反)
新增 `tests/invariants.py`,對每個輸出 JSON 跑:
- `diagonal_count == 0`(strict Manhattan 已是現況)
- `no zero-length segment`
- `every endpoint coord finite & inside image bbox`
- `every door/window endpoint MUST satisfy: coincides with a wall endpoint, OR lies on a wall body within ε, OR (新)在牆 mask 上連續可達 wall endpoint within K`
- `no segment with NaN / inf`

invariant 違反 = test FAIL,沒有容差。

### 18.2 — Metric 計算器
新增 `tests/metrics.py`:輸入 JSON + 原始 BGR,輸出
```python
{
  "wall_iou": float,            # rasterize walls → IOU vs wall mask
  "opening_iou_door": float,    # 同上,door
  "opening_iou_window": float,  # 同上,window
  "floating_openings": int,     # 不滿足 invariant 中「最低 1 端錨在牆」的 opening 數
  "free_endpoints": int,        # degree-1 節點數(用統一精度,見 §4)
  "phantom_wall_px": int,       # 牆段壓在 wall_evidence < 0.3 區域的像素數
  "segment_count": int,         # 監看用,不 gate
  "endpoint_degree_hist": dict, # 監看用
}
```

Rasterization 用 `cv2.line` thickness=1(或對應 local_thickness),保證可重現。

### 18.3 — Regression 規則改寫
[regression.py](regression.py) 把 byte-equal 換成:
- 讀 `tests/baseline/<case>_metrics.json`
- 跑 pipeline 算 current metrics
- 對每個 metric 套 per-metric 容差(IOU ±0.005、count ±0)
- 跑 invariants;任一違反就 FAIL
- 任一指標在容差外變差 → FAIL;全部在容差內或變好 → PASS

舊 `tests/baseline/<case>.json` 仍保留供 audit/debug,但**不再 gate**。

### 18.4 — Visual diff 工具
`tests/diff_view.py`:吃 baseline JSON + current JSON + 原圖,產三聯圖(baseline / current / 紅綠 overlay)。當 metric 在容差邊緣或 user-approve 對話需要視覺判斷時用。

### 18.5 — Baseline migration
跑一次 current pipeline → 把 metric 結果存成新 baseline。所有後續 step 都對這份 metric baseline 比較。

**Done when**:`pytest tests/` 跑舊 bit-identical 全綠的同時,也跑新 metric-based 全綠;某個刻意製造的「拆牆」測試會在 metric-based 上 FAIL、bit-identical 上也 FAIL(雙重防護驗證)。

---

## §2 · Step 19 — 源頭修法:Skeletonize 前先填色塊洞(✗ 試 + revert)

**結果**:試了單純的 `wall_for_skel = wall ∪ dilated(chromatic)` 路徑,**反向**。

- 牆中線確實連續了,但 output 多出穿過門窗的牆段
- source: `wall_iou_vs_source` 0.4569 → 0.3818;total_length +32.5%
- sg2: 全部 IOU 下降(wall −0.025、door −0.015、window −0.014)
- 渲染上門窗被牆覆蓋,**明顯變差**(triptych 確認)
- 主因:現況 step 17 trunk_split 已經把 source/sg2 的 floating_openings 收到 0;§19 在解決一個已經沒了的問題,代價是污染牆 IOU

**結論**:若要走這條路,必須配套「chromatic-subtract 後處理」(把牆段在 chromatic body 上的部分切掉),但那等於把牆又切回片段,§19 的「連續性」好處被吃掉。改走 §20 的針對性修法。

關鍵 learning:在動源頭之前,先看 metric baseline ——
若 floating_openings 已經是 0,§19 沒空間;只剩 Gemini 還有 3 個 → 走 §20。

---

## §3 · Step 20 — End-of-pipeline invariant:「色塊端點必須錨在牆」

**動機**:§18 metric 層做完後,floating_openings 變數實際分佈:
- source: 0(step 17 trunk_split 已解決)
- sg2: 0(同上)
- Gemini: 3(仍是 GOAL violation,需要這層補上)

§20 把 GOAL invariant 升級為 STRICT,並提供一個 candidate generator 主動處理現有 GOAL violation。這層讓需求 (b) 從 "best effort" 變成 **post-condition**。

### 20.1 — 新 candidate generator `chromatic_anchor_candidates`
插在 `trunk_split` 之後(現在是 pipeline 最後)。對每個 door/window 端點檢查 — 若無法通過 invariant.check_openings_anchored 則嘗試修補:
1. 找最近的牆端點(在 max_radius 內、與 chromatic 段共享一個 near-axis)
2. 若兩者間 wall mask 有連續證據(support ≥ K%):emit 候選 — 插入新 wall segment 從 chromatic endpoint 通到該 wall endpoint
3. 沒有 mask 支持:不 emit(可能該 opening 本身就是誤偵測,留給 audit log)

跟 `proximal_bridge` 的差別:proximal_bridge 只配對 wall↔wall;這個新 generator 是 chromatic→wall 的非對稱配對,專門收 chromatic invariant violation。

### 20.2 — 用 invariant 升級為 STRICT 強制執行
[tests/invariants.py](tests/invariants.py) 的 `floating_opening` 目前在 GOAL 層,在 §20 完成後從 GOAL 移到 STRICT。任一 chromatic endpoint 仍 floating → regression FAIL。

### 20.3 — 處理「真的就是錯誤偵測」的情境
若 generator 跑完仍有 floating endpoint(沒任何 wall 在附近 anchorable),代表色塊偵測在白底中誤報。短期:留為 STRICT FAIL,使用者得手動處理。中期:在 step 1 color segmentation 加 chromatic isolation 過濾(沒被牆包圍的孤立色塊丟掉)。

**Done when**:`floating_openings == 0` 是 invariant 而非統計;三張 reference image 都通過(Gemini 重新解封)。

---

## §4 · Step 21 — 統一 degree-counting 精度

**動機**:現在四個地方各自做 degree counting,精度不一致:
- [vectorize.py:715](vectorize.py#L715) `_build_degree_map` 用 `quantize=0.01`
- [generators.py:1782](generators.py#L1782) `trunk_split_candidates` 用 `int(round(...))`
- `audit_view._free_endpoints` / `scoring._node_degree_counter` 各自一套

不同 pass 對「兩個端點算不算同一點」有不同答案,是隱性 bug 表面。

### 21.1 — 新 `geom_utils.py`
集中:
```python
ENDPOINT_PRECISION = 1e-4  # 與 _accept_2d_cluster_candidates 的 round(_, 4) 對齊

def endpoint_key(x, y): return (round(x, 4), round(y, 4))
def node_degree(segments): ...
def free_endpoints(segments): ...
```

### 21.2 — 全模組改 import
四處 call site 全部改 import `geom_utils.endpoint_key`;移除各自的 `int(round)` / `quantize=0.01` 私有邏輯。

**Done when**:四個 free-endpoint 統計值在同一份 JSON 上完全一致;metric-based regression 全綠。

---

## §5 · Step 22 — Score 加連續信號 + 撤掉 `skip_score=True`

**動機**:現況 `parallel_merge` 與 `fuse_close_endpoints` 永久 `skip_score=True`(step 15 已量化結論)。根因是 score 的 free_endpoint / junction / pseudo_junction 都是整數 count,sub-pixel 修正在整數狀態之間沒有梯度,score 看不到「越來越近」這個過程。

### 22.1 — 加 `opening_body_attachment` 連續項
[scoring.py](scoring.py):每個 door/window 端點到最近**牆身**(取線上最近點,不只端點)的距離 `d`,計分:
```python
score += sum(exp(-d / tau) for endpoint in opening_endpoints)
```
`tau` 約等於 wall thickness。離牆遠 → 接近 0;落在牆上 → 接近 1。提供 sub-pixel 梯度。

### 22.2 — 加 `opening_phantom` 對稱負項
類比 [scoring.py:533](scoring.py#L533) `phantom_penalty`(牆壓在白底):門/窗段壓在色塊 mask 證據低處的負分。讓 score 主動排斥孤立色塊。

### 22.3 — 把 free_endpoint / pseudo_junction 改連續
不是「count」,改成「pressure」:
```python
free_endpoint_pressure = sum(1 / (1 + min_dist_to_nearest_other_endpoint))
```
整數狀態之間有梯度。同樣處理 `pseudo_junction`(目前只算 degree-1 在牆上,改成連續衰減)。

### 22.4 — 重跑 skip_score=True 移除實驗
有了 §22.1-3,試把 `parallel_merge` / `fuse` 的 `skip_score=True` 撤掉:
- metric-based regression 全綠 → 撤掉。
- 還是退化 → 用 audit_view 查 reject 原因、寫具體報告、保留 skip_score=True 並更新 [vectorize.py:1093](vectorize.py#L1093) 的 Score-gate policy 框架說明。

**Done when**:至少 `_accept_fuse_candidates` 撤掉 `skip_score=True` 仍 metric-PASS;若 `parallel_merge` 撤不掉,有量化證據說明為什麼。

---

## §6 · Step 23 — 收斂 17 道線性 cascade 成單一 accept-loop

**動機**:現況 [vectorize.py:2373-2684](vectorize.py#L2373-L2684) 是 17 道順序敏感的 pass。每一道都假設前面跑了什麼、後面會修什麼。順序變一下就漂。Generator 已經就緒(step 7-10 完成),差最後一里。

### 23.1 — 候選串並列化
寫一個 `master_accept_loop(segments, generators, score_fn)`:
- 每輪呼叫所有 generator 產生候選池
- 用 score 對全池排序
- 貪婪接受最高分候選、locked endpoints、重新生成
- 沒有候選通過 score gate → 收斂

不再有「t_project 必須在 fuse 之後」這種隱性順序假設;順序由 score 動態決定。

### 23.2 — 保留兩個 phase boundary
- **Phase A**:純幾何修正(axis_align、manhattan_force、canonicalize)。這幾個有「無條件 transform」性質、不需要 score。一次跑完。
- **Phase B**:master_accept_loop,所有 candidate-based generator 並列。

Skeletonize → branches → Phase A → Phase B → trunk_split → chromatic_anchor invariant。整條 pipeline 從 17 步變 5 步。

### 23.3 — 用 audit log 驗證
metric-based regression 全綠;audit log 看 generator 接受比例分佈、確認沒有 generator 完全 starve。

**Done when**:`vectorize_bgr` 函式可讀性顯著上升,移除 12+ 個 `_accept_*` wrapper 函式或合併成 master loop 的 hook。

---

## §7 · Optional · Step 24 — 移除 `compute_local_thickness` 對 wall mask 的依賴

目前 [canonical_line.py](canonical_line.py) 算 thickness 要 wall mask。若 §2 完成後 wall_for_skel 是連續的,thickness 可以從 skeleton 與 mask 距離直接算。架構乾淨化、Phase A/B 切分更明確。

---

## 進度狀態

| Step | 名稱 | 狀態 | Owner |
|---|---|---|---|
| 18 | Metric-based regression | ⬜ Not started | — |
| 19 | Skeletonize fills chromatic holes | ⬜ Not started(blocked on 18) | — |
| 20 | Chromatic-anchor invariant | ⬜ Not started(blocked on 18) | — |
| 21 | Unified degree precision | ⬜ Not started | — |
| 22 | Continuous score signals | ⬜ Not started(blocked on 18) | — |
| 23 | Single accept-loop | ⬜ Not started(blocked on 22) | — |
| 24 | Thickness without wall mask | ⬜ Optional | — |

---

## 工程紀律(避免改 A 壞 B 復發)

1. **動 pipeline 前先動 §1**。沒有 metric-based 規則就不要碰 Step 19+;否則回到「靠 byte-equal 維持秩序」的舊循環。
2. **每個 step 一個 commit、一次 metric-regression run**。commit 訊息附 metric diff 表。
3. **發現某個 step 需要拆 case**:停下來、問自己是不是落回 case-specific 補丁、寫進 audit 紀錄。
4. **`skip_score=True` 新增禁止**。step 11/15 已經分析過為什麼現有的不能撤;新加的 generator 一律走 score gate。如果非要,要在 §22 框架下重新做 audit 驗證。
5. **每個新 generator 寫成純函式**:輸入 (segments, masks, opts) → 輸出 List[Candidate]。不讀檔、不寫檔、不依賴順序。
