# TRAINING_REPORT_ITER2 — CES-PPO 迭代2：纯估计器隔离重训 A1′+A2′

**实验**：exp_4ce018a2（阶段2b·步骤6·迭代2，SPEC v1.1）｜**日期**：2026-11-15｜**仓库**：`yumxcode/g1-locomotion-train` 分支 `ces-ppo` @ `ae3e5e8`
**判决**：**falsified（再次证否）——事件分段优势估计器包本身（与奖励无关）压制步态涌现；事件对齐与随机切分不可区分。**

---

## 1. 问题与单变量设计

迭代1（exp_d03e0ea2，已定案）：CES-PPO A1（事件分段）/A2（随机边界）在 v6 环境 + w_loc=0.05 足局部奖励下双双崩坏；判崩主因为 r_imp（切换次罚 ~0.17 vs 迈步收益 +0.02/步），但无法排除估计器包的独立贡献。

迭代2 剥离全部足局部奖励做单变量隔离：

- **A1′**：v6 奖励**逐字不变**（w_loc=0，实现为奖励表无 `ces_loc` 键）+ 仅启用估计器包：事件分段（去抖 L_min=5）、段内 TD(λ=0.95)+段边界硬 bootstrap、段内优势归一化、整段 minibatch（全局 batch 不变）。**对 A0 的唯一差异 = 优势估计器包。**
- **A2′**：同 A1′，唯一差异 = 段边界随机放置（段长 multiset 保持，`shuffle_segment_boundaries`）。
- 其余全部继承 v6：网络（MLP 128×64 actor/critic）、entropy 0.01、KL 自适应 lr、PD 60/1.5、域随机化、78 维观测、**seed=1（与 v6 配对）**。

实现要点（commit `ae3e5e8`）：基类 `_prepare_reward_function` 会 pop scale==0 的奖励项（连函数都不构建），故 w_loc=0 不能用 `ces_loc=0.0` 表达——那会连分段导出一起跳过。因此新任务 `g1_23dof_ces2`：奖励表不含 `ces_loc`（v6 逐字），CES 环境子类在 `compute_reward()` 之后调用 `_ces_step()`（原迭代1奖励函数拆出的分段+导出逻辑），读取的张量状态与迭代1调用点逐位相同。奖励流=v6、估计器流=迭代1，两者解耦。

## 2. 运行与成本台账（gradmotion，镜像 BJX00000001 = IsaacGym preview-4）

账号：48 (ridovi5230) 余额不足（run 报"账户余额不足"）→ 按授权轮换 **49 (ketin84997, userId 4427)**，新建项目 PRO_20261115_002；goodsId=ESKU000001（1×4090D 24G，**¥5.76/h**）。注：49 下该 goods 不支持 /personal 挂载（错误码 601），已取消挂载（输出走 logs/+output/ 通道，无影响）。

| 任务 | taskId | 用途 | start–end | 时长 | 费用(¥) |
|---|---|---|---|---|---|
| ces2-a1p-smoke-200it | TASK_20261115_115 | 管线验证（200 it） | 12:34:52–12:40:40 | 5.8 min | ≈0.56 |
| **ces2-a1p-10k（A1′）** | **TASK_20261115_116** | 事件分段臂（10000 it 预算，it2500 止损） | 12:43:39–13:15:18 | 31.6 min | ≈3.04 |
| **ces2-a2p-10k（A2′）** | **TASK_20261115_117** | 随机边界臂（同上，串行） | 13:43:18–14:14:47 | 31.5 min | ≈3.02 |

合计 ≈ **¥6.6**（预算 ≤¥30 ✓）；GPU 各 0.53 h（预算 ≤2.5 h/臂 ✓）；单卡串行执行（pit_9e74a116 纪律）。种子：全部 seed=1。速度 ~0.76 s/it（A1′ 2500 it 用 26 min）。

## 3. 生效性验证（运行时证据）

- **奖励=v6 逐字**：平台 data keys 的 `Episode/rew_*` 共 16 项与 v6 完全一致、**无 rew_ces_loc**；训练日志 `Mean episode rew_*` 分量列表同样 16 项（无 ces_loc）。（smoke 任务另有 `effective reward scales` 启动打印。）
- **估计器在跑**：两臂每 iter 打印 CES segment stats（A1′ 末 iter n=15240 段、len_median=5=L_min 去抖生效；A2′ n=15670、median=5；shuffle 前后段长统计一致 = 随机化正确性运行时证据）。诊断分量（slip/imp/duty raw 均值）照常计算——**仅诊断，不进奖励**（duty_credit≡0，re-spread 数值无操作，本地测试 A1 项验证 bit-identical）。
- 本地测试：回归 24（test_ces）+9（integration）+ 新增 15（test_ces2：零 credit 无操作、estimator-only 端到端 a1/a2、multiset 保持）全部 PASS。

## 4. 结果：三臂同协议评估（4096 episodes，v6 同款 stage2_eval）

详见同目录 `EVAL_TABLE_ITER2.md`。核心行：

| | A0=v6 | A1′ 事件分段 | A2′ 随机边界 |
|---|---|---|---|
| RMSE (m/s) | **0.407** | 0.803 | 0.817 |
| no-fall | **99.90%** | 0% | 0.02% |
| ep_len 轨迹 | it1000:46 → **it2000:297** → it3000:712 | 平台 36-42，it2500=41.9 | 平台 38-42，it2500=38.9 |
| it2500 止损门 | n/a | **触发（41.9≤50 → STOP）** | **触发（38.9≤50 → STOP）** |
| noise std 终值 | 1.69 | **0.637（收敛）** | **0.679（收敛）** |
| 段长统计 | n/a | median 5（=L_min），mean 6.45 | median 5，mean 6.27 |
| jit / onnx 一致性 | 0.0 / 5.7e-6 PASS | **0.0 / 4.29e-6 PASS** | **0.0 / 4.29e-6 PASS** |

**it2500 早期止损判据执行记录**（两臂均触发，日志行在案）：
- A1′：`[gm] EARLY-STOP GATE iter 2500: mean ep_len (last 100 eps) = 41.92 (v6 reference: it2000 ~297) floor=50.0 -> STOP ARM` → 保存 model_2500.pt、照常评估+导出、任务正常结束（st=5）
- A2′：`… = 38.94 … -> STOP ARM`（同上）
- 两臂均触发 → **整体止损**（不再自费加跑，符合预注册分支 1）。

checkpoint/导出：两臂最优（止损点）checkpoint model_2500.pt 已下载（`ces2_results/a{1,2}p/checkpoints/`）；jit/onnx 导出与 <1e-5 前向一致性由平台 gm_run_all stage3 自动执行并通过（数字见上表；导出件 policy_1.pt/.onnx 在两任务 output/ 通道）。

## 5. 与迭代1对照：剥离局部奖励后步态恢复涌现了吗？

**没有。** 迭代1 双臂崩坏（RMSE 0.92/0.96、no-fall 0%、平台 32-48、noise std 发散 10.2/3.56）；迭代2 剥离 r_imp/slip/duty 全部足局部奖励后，两臂**仍停在 36-42 平台、未复现 v6 it2000≈297 的爆发**，且 it2500 触发预注册止损。因果链修正为：

1. **估计器包独立致崩**：奖励与 v6 逐字一致时（对 A0 唯一差异=估计器包），步态涌现完全被压制。迭代1 的"奖励结构主导"结论需要修正——**奖励结构与估计器包两者皆致命**：迭代1 = 两者叠加（且 r_imp 触发熵发散型崩坏）；迭代2 = 纯估计器（平台型压制，无熵发散）。
2. **失败模式变化（重要诊断）**：迭代1 noise std 发散至 10.2/3.56（r_imp 使迈步净收益≈0，无强优势信号→熵系数 0.01 推 std 无界）；迭代2 noise std **收敛**至 0.637/0.679，torque/dof_acc 分量降至 v6 行走态的 1/30（-0.0014 vs -0.046）——策略平稳收敛到"近静止不迈步"的局部最优。估计器包的压制机制 ≠ 熵发散，而是**优势信号被切碎/归一化后，跨段（跨接触事件）的长期回报差异不再有效引导迈步行为**。
3. 段长统计：两臂 median=5（=去抖 L_min），接触在崩溃前期本就频繁切换——事件边界密集时整段 minibatch + 段内归一化会进一步降低有效信用分配粒度。

## 6. A1′ vs A2′ 差异分析：估计器对齐有可区分的收益吗？

**没有。** 事件对齐（A1′）与随机切分（A2′）在所有指标上同水平：ep_len 41.9 vs 38.9（差 3.0，各臂自身波动 ±3 内）、RMSE 0.803 vs 0.817、no-fall 0% vs 0.02%、noise std 0.637 vs 0.679、能耗代理同量级。**若"事件对齐的分段结构"携带真实信息，A1′ 应优于 A2′——数据不支持**。与迭代1 的 A1 vs A2（0.921 vs 0.956，同样不可区分）一致。

## 7. 诚实结论

- **核心新颖性主张证否**：在此任务配置（23dof G1、契约 v0、无步态时钟、冻结 v6 奖励、PPO 超参不变）下，事件分段优势估计器包（含段内 TD+硬 bootstrap、段内归一化、整段 minibatch）**既不优于 A0（RMSE 0.80 vs 0.407），也与随机切分不可区分**——中性都不成立，是有害的。迭代1+迭代2 合并结论：CES-PPO 当前形态在此配置下不可用，两个成分（足局部奖励、估计器包）各自独立致崩。
- **局限**：(a) 两臂按预注册判据于 it2500 止损（2500/10000 预算）——v6 涌现在 it2000 前已明确且两臂末 1500 it 零上升趋势，晚爆发可能性极低，但严格说未观测 it2500 后行为；(b) 单 seed（seed=1，与 v6 配对设计）；(c) 估计器包是六组件捆绑，本轮未拆分定位具体致崩组件（候选：整段 minibatch 打破样本独立性 + 段内归一化在密集短段下的信号损失）。
- **若重访**：先做组件消融（只换 GAE→分段 TD、保留标准 minibatch）与段长下限提高（L_min≫5，或按步数而非事件分段）；机制假设「整段 minibatch 的样本相关结构 + 短段归一化削减长期 credit assignment」可由 it500-2500 的 value loss 曲线与优势分布直接检验。

## 8. 产物清单

- `EVAL_TABLE_ITER2.md` — 三臂同协议评估表（本目录）
- `a1p/` — A1′：a1p_final_summary.json（eval/export/early_stop 原始摘要）、checkpoints/model_2500.pt、curves/（ep_len、mean_reward、noise_std CSV ×3）、tfevents_logs.tgz
- `a2p/` — A2′：同上结构
- 代码：`ces-ppo@ae3e5e8`（g1_23dof_ces2_env/config、gm_run_all 内置 it2500 门控、tests/test_ces2.py）
- gm 任务笔记已含全部 taskId 与成本（§2 台账）
