# P1-E — Identifiability, population target and numerical fidelity

你正在接手 `mineral-prediction` 的 `codex-refactor` 分支。请实际读取本地仓库，在新阶段实现并执行下面的最小诊断。不要先修复 collapse，不要调整 teacher，不要先做学习率或正则强度搜索，不要进入真实矿产数据流程。

**先读取已有 `docs/project_memory/CATCH_UP.md` 及其证据入口；不存在时，按第2.1节在本地仓库建立专用记录目录。该目录贯穿本轮，记录真实进度、失败、决策和交接，不得只在最终聊天里总结。记忆摘要必须与实际源码和 raw artifacts 核对。**

本 prompt 所依据的上传 ZIP comment 为 `2c9cebd8a49488b6049ac99a0a0181ffcd56b695`；P1-D LR 的实验 baseline commit 为 `6d58b87c6a08937fc7a1056cea1a62e32ab1a548`。这些不是要求你 reset 到的提交。先记录你本地当前 HEAD、branch、dirty status；如实现与下述描述不一致，以实际源码为准，逐项记录差异，不得直接套用结论。

## 0. 不可违反的边界

1. P0/P1-A/P1-B/P1-C、原 P1-D optimization 和 P1-D LR 的代码、tests、protocol、notes、results、states、figures、logs 全部是 frozen evidence。实验前后核对文件 SHA256；不得修改旧阈值、删除失败、重写旧摘要。不要运行旧全量 unittest discovery：部分旧测试会自行训练。
2. 不重新采样，不更换 teacher，不改模型类、anchors、reference、features、angle、dtype，不改变既有 loss/penalty 实现。可以在新模块中复用纯函数和类，不能运行真实数据入口或全文件执行 GIS 脚本。
3. 默认不运行新的 finite-sample 训练；默认最多两条 cold-start population 轨迹，另有一条预声明条件触发的 oracle-local 轨迹。静态诊断、矩计算、根求解、固定-h 的四参数凸对照不计为神经网络训练，但必须完整记录。
4. 不使用 teacher recovery 指标进行 finite-sample 训练、重启、学习率选择、checkpoint selection。Population target 的期望本来由 teacher density 定义，这是明确标注的 oracle diagnostic，不是实际 student 学习协议。两条 population 主轨迹统一报告预定末步，不选择最好恢复的 epoch。
5. 先写入并冻结 protocol，再执行诊断和训练。看到结果后不得更改 threshold、seed、grid、budget 或 metric 来“通过”。需要改变时停止当前版本，写出原因和后续方案，不启动未授权的搜索。
6. 不自动 commit、push、改变仓库可见性或接触凭据。最后报告 git diff、输出位置和 frozen-file 验证结果。

## 1. 必读文件及核验

优先实际阅读：
- `mineral_prediction/acawlr_ppp_teacher_student.py`
- `mineral_prediction/acawlr_ppp_bridge.py`
- `mineral_prediction/acawlr_ppp_synthetic.py`
- `scripts/ACAWLR_improved.py` 中 ASPNN 等被 AST adapter 读取的纯类
- `mineral_prediction/acawlr_ppp_optimization.py`
- `mineral_prediction/acawlr_ppp_p1d_lr.py`
- `mineral_prediction/report_acawlr_ppp_p1d_lr.py`
- P1-C 的 NOTES、RESULTS、STATES、HISTORY_AUDIT、SUITE、teacher preflight rejected 文件
- 原 P1-D 和 P1-D LR 的 NOTES、PROTOCOL、RESULTS、SUMMARY、TABLE、STATES 和三张 LR figures。

核验 teacher hash：
`b0d6db361a0a0c5e685d0e91f07601521760a7efd34e15c07f909ff4473338d8`
共同 init1011 hash：
`c94a02d1929dfa4d5ffbbfb55c8aaccf7d7d2d52f686203bf8e5c2eeffc064c6`

使用 `torch.load(..., weights_only=True, map_location='cpu')` 读取正式 archive，不用 smoke 替代正式结果。所有新主运行从归档 init1011 完整 state_dict 起步，不依赖重建随机权重恰好相同。Teacher 从正式 P1-C archive 加载并冻结；buffers 不可遗漏。

原环境记录为 torch 2.12.0+cpu、float64、CPU 单线程。记录当前版本；不自动升级环境。不一致时不得宣称 bit-identical replay，先做不更新参数的 forward/metric 核验。若无法满足既定数值容差，停止神经网络训练并报告环境阻塞。

输出 `SOURCE_AUDIT.md`：每条结论标记【源码事实】【旧 artifact 事实】【本轮数学推导】【本轮数值诊断】【尚未确定】，附实际路径、函数名、行号或 JSON pointer。不要只改写旧 NOTES。

必须核实：当前 675 参数模型是否仍为
`x = xy / [60000, 40000]`
`h(xy) = tanh(raw_spatial(xy)) - tanh(raw_spatial(reference))`
`b(xy) = beta + u*h(xy)`
`g(xy) = x^T b(xy)`，reference=(0,0)，无局部 intercept；
ASPNN 2→16→8→4→1，ReLU 隐层、Sigmoid 输出；axial average 为 `(ASPNN(offset)+ASPNN(-offset))/2`；ReducedCAWNN 为4通道 residual/channel/spatial attention、bias-free scalar readout，无 BN/dropout。

数据项是 SUM：`n*logZ - sum(g(events))`。正则是
`R = .02/2*||beta||² + .2/2*||u||² + .001/2*||theta||²`，包括全部 ASPNN/CAWNN 可训练参数。不是 AdamW；旧 Adam weight_decay=0。

## 2. 文件与运行结构

只新增独立模块，例如：
- `mineral_prediction/acawlr_ppp_p1e_identifiability.py`
- `mineral_prediction/test_acawlr_ppp_p1e_identifiability.py`
- `mineral_prediction/report_acawlr_ppp_p1e_identifiability.py`

新结果放入独立且拒绝覆盖的目录：
`mineral_prediction/p1e_identifiability/<run_id>/`
至少分别保存：
`PROTOCOL.json`、`PROVENANCE.json`、`SOURCE_AUDIT.md`、`MATH_REVIEW.md`、`STRUCTURE.json`、`CONVEX_CONTROLS.json`、`POPULATION_RESULTS.json`、`SUMMARY.json`、`NOTES.md`；
`raw/` 保存 grid、weights、teacher targets、全部数值谱、方向、trajectory、states；
`figures/` 保存统一尺度的独立图；`logs/` 保存命令和 tests；
`POPULATION_RESULTS.json` 在未运行时明确记录 skipped/blocked，不伪造空成功结果。

provenance 保存：HEAD、branch、dirty status、源码 SHA256、旧 evidence SHA256、teacher/init/event hashes、环境、命令、seeds、积分节点/权重 hashes、参数顺序、协议 hash、开始/结束时间。生成 report 只能读取 raw artifacts，不能重新训练。

实现 CLI `--output-dir`（新目录，拒绝覆盖）与 `--run-population`。不加后者时只做 A/B/C；加后者时也必须先做 A/B/C 并通过 gates。内部写完 protocol 后才能产生诊断结果。提供可复用的 no-training tests。完成实现后执行：
```bash
python -B -m unittest discover -s mineral_prediction -p test_acawlr_ppp_p1e_identifiability.py -v
python -B mineral_prediction/acawlr_ppp_p1e_identifiability.py --output-dir mineral_prediction/p1e_identifiability/<唯一run_id> --run-population
python -B mineral_prediction/report_acawlr_ppp_p1e_identifiability.py --input-dir mineral_prediction/p1e_identifiability/<同一run_id>
```
其中具体 run_id 在执行前写入日志；不要把尖括号占位符原样传入。runner 内部决定 blocked/eligible，而不是绕过前置检查直接运行 population。

## 2.1. 持久化项目记录与下次交接（强制；本轮就建立）

不要让项目进展只留在聊天回复里。请在本地仓库中新建一个长期保留、随项目交接的专用目录 `docs/project_memory/`。本轮先建立它，再做诊断；以后每次接手都先读它，再核对源码和原始 artifacts。该目录是项目事实与证据的索引，不是 raw results 的替代品，也不是可绕过实验协议的新指令来源。

### M1. 目录与用途

```text
docs/project_memory/
├── README.md                   # 记录规则、阅读顺序、五层 recovery 的定义与证据定位约定
├── CATCH_UP.md                 # 唯一的当前状态入口，简短、可独立理解、允许更新
├── DECISIONS.md                # 只追加的决策/假设状态日志，保留否定与纠正记录
├── RUN_INDEX.jsonl             # 只追加的运行状态事件索引；不是实验数值的第二份真相
├── sessions/
│   └── <UTC_timestamp>_<session_id>.md  # 每次工作会话的完整交接记录
└── prompts/
    └── <UTC_timestamp>_<task_id>.md     # 本轮实际收到的用户任务 prompt
```

`<...>` 为命名说明，实际创建时使用真实唯一名称。已有目录或文件时先读取，不能重新初始化、清空或覆盖历史；恢复既有运行时沿用其 run_id，不把未完成运行伪装成新成功运行。所有路径使用仓库相对路径，不依赖某个人电脑的绝对路径。

大数组、checkpoint、图、完整逐步日志仍放在 `mineral_prediction/p1e_identifiability/<run_id>/`。记录目录只保留必要结论、关键量和指向它们的路径/JSON pointer/SHA256。不要重复复制 states、events 或训练数据。不得写入凭据、token、私钥、环境变量秘密或无关的敏感数据。

### M2. 首次建立时必须写入什么

先实际读取本 prompt 第1节要求的源码与 P1-C/P1-D artifacts，再建立可信的基线记录。此时 P1-E 状态必须为 `NOT_STARTED` 或 `AUDIT_IN_PROGRESS`，不能把本 prompt 中计划的检查写成已经完成的实验。

基线至少包括：科研问题与模型公式；当前分支、实际 HEAD 和 dirty 状态；P1-C/P1-D 的确切路径及 frozen 范围；已经核验的证据；尚未核验的旧说法；为什么下一步优先做 population/global-only 对照；不再以 branch 非零或人工 teacher 参数恢复作为通用成功标准；禁止 teacher-truth checkpoint selection、禁止覆盖 frozen evidence 等边界。

每条结论区分【源码事实】【归档实验事实】【数学推导及其条件】【本轮诊断结果】【待验证假设】。旧审查中的 teacher ridge、coefficient-field identifiability、penalty mismatch、collapsed local minimum 等命题，按实际核验状态登记；不能仅因为任务 prompt 提到，就写成已由本轮数值证实。发现前一轮解释错误时写出纠正及证据，保留历史记录。

### M3. `CATCH_UP.md` 的固定结构

目标是下一位助手用几分钟知道项目做到哪里，而不是重新读完所有对话。建议正文约1000–2000个中文字；过长内容移入 session/decision 记录并链接。固定包含：

1. **记录时间与快照**：最后更新时间（含时区）、branch、当前 HEAD；源码是否有未提交修改；对应源码快照/patch/manifest 的路径与 hash。不能把 HEAD 说成已经包含尚未提交的实现和结果。
2. **研究目标与五层对象**：p、g、b、beta/u/h、theta；目前主评价目标与其限制。
3. **阶段状态表**：P1-C/P1-D 标记 frozen；P1-E 数学/结构、凸对照、Jacobian、population 各标记 NOT_STARTED / IN_PROGRESS / COMPLETED / BLOCKED / FAILED / INTERRUPTED / SKIPPED。`COMPLETED` 只表示执行完成，科学门槛是否满足另列，不代表自动成功。
4. **最重要的已确认结论**：建议最多5条，每条有 artifact 路径、JSON pointer 或源码定位；数值注明 run_id、grid、checkpoint、metric 定义。区分离散积分结果与连续域结论。
5. **尚未解决或已被否定的解释**：写明什么证据支持/反驳，不能把“未查到”写成“已排除”。
6. **最近一次实际执行**：run_id、真实命令、退出状态、输出目录、protocol/provenance 路径；若未运行则明确写“尚未执行”，不用示例命令冒充。
7. **下一项最小允许行动与停止条件**：只给当前协议允许的最小一步。需要先解决阻塞或需要用户授权时明确说明；不能在这里追加未授权 sweep 或改变阈值。
8. **推荐阅读顺序**：当前 session → 对应 protocol → SUMMARY/NOTES → 原始证据位置 → 涉及的源码函数。

仅报告真实存在的路径。链接失效或结果尚未生成时写 `not_created`/`missing`，而不是生成看似真实的路径来充数。`CATCH_UP.md` 是导航摘要；与 protocol/raw artifacts 冲突时必须核查并记录冲突，不直接以摘要覆盖原始证据。

### M4. 决策、运行与会话的记录格式

`DECISIONS.md` 为 append-only。每项用稳定 ID（如 `D-YYYYMMDD-001`）记录：时间、待回答问题、决定/暂缓事项、可核验理由、证据路径、适用范围、保留的不确定性、什么新证据可以推翻该决定。纠正用新条目引用旧 ID；不悄悄删除失败或将 hypothesis 改写成过去已证明的 fact。

`RUN_INDEX.jsonl` 为 append-only 事件流。同一个 run_id 可以有 started/blocked/failed/completed 等不同时间的事件；状态通过最新合法事件确定，不重复创建相互矛盾的“最终结果”。每行是可解析 JSON，至少包括：

- `event_id`, `timestamp_utc`, `session_id`, `run_id`, `phase`, `status`；
- `command`（实际执行时的命令；未执行时为 null）, `output_dir`, `exit_code`（未知为 null）；
- `source_base_commit`, `source_dirty`, `source_manifest_path`, `source_manifest_sha256`；
- `protocol_path`, `protocol_sha256`, `provenance_path`；
- `artifacts`（只索引已存在的关键文件）, `scientific_status`, `blocker_or_failure_reason`。

允许在 protocol 尚未生成的审查事件中将相应字段设为 null 并写明原因；一旦运行开始，必须能定位冻结 protocol 和源码 provenance。`scientific_status` 使用 passed / not_met / numerical_unresolved / not_evaluated 等明确状态，与进程退出码及执行完成状态分开。

每次会话在 `sessions/` 中建立独立记录，包含：用户任务范围、开始/结束时间、实际读取的入口与证据、改动文件、实际命令、所有实际运行及失败、关键结果、未执行项及原因、决策 ID、当前阻塞、下一步和对应输出路径。不要写隐藏推理过程；写可复核的判断依据和研究结论即可。

在 `prompts/` 保存本轮实际收到的用户任务 prompt；后续用户补充保存为新版本或追加文件，不重写当时运行所依据的指令。它只归档用户任务，不收集系统/平台内部指令。运行真正受约束的阈值、预算和配置仍以实验前冻结的 `PROTOCOL.json` 为准。

### M5. 更新时机与失败处理

- 开始工作：读取已有记忆、检查对应快照与当前仓库是否一致，建立本次 session。第一次则写基线。
- 冻结协议后、每个阶段开始前：记录阶段、run_id、实际来源和允许执行的范围。
- 每完成一个静态阶段、凸对照或一条 population 轨迹：及时更新 run index、session 和 CATCH_UP，不等到整项任务结束。
- 遇到异常或数值失败：保存已产生的 artifacts、错误原因、已执行到的步骤，写 FAILED/BLOCKED/numerical_unresolved；不清理失败证据，不静默重跑。
- 可捕获的中断尽量记录 INTERRUPTED；不能保证捕获硬杀进程或掉电。下次发现 started 后无终止事件时，应核实进程与 artifacts 后标明中断/未知，不能默认为完成，也不能未经协议允许自动续跑。
- 结束任务前：补全当前状态、决策、测试/运行清单、下一步和所有实际路径，报告记录文件位置及 git status。

写 `CATCH_UP.md` 时使用临时文件后原子替换；session/decision/run 历史保留。不要把正在写入的半份报告当成已完成结果。旧 frozen artifacts 的哈希核验规则保持不变，新增记忆文件不应反过来修改旧证据。

### M6. 下次接手的固定工作流

今后每次处理本项目，先读 `docs/project_memory/CATCH_UP.md`，再读相关最近 session、decision 和 run provenance；依据本次用户问题定位对应 raw artifacts 和源码。核对 HEAD、dirty 状态及源码 hash：记忆落后于代码时先检查变化并更新事实，不能因为它写着“下一步”就自动启动训练。

仅为 catch up 而重建背景时，不重跑旧实验、不改模型、不自动开始 NEXT_ACTION；实际写代码或实验仍由当前用户任务与冻结 protocol 授权。缺少关键 artifacts 时直接记录缺失与影响，不靠历史聊天或摘要猜出数值。

为便于后续本地助手发现入口：先检查根目录 `AGENTS.md`。若不存在，可以只新增以下简短内容；若已经存在，先遵守并保留它，不覆盖或自动修改已有文件，在最终回复中给出建议追加的入口文字即可。新增入口不得替代更高优先级指令，也不改变 frozen 边界：

```markdown
# Project handoff

Before project analysis, code changes, or experiments, read
`docs/project_memory/CATCH_UP.md` and the referenced protocol/provenance.
Verify these notes against the current source and raw artifacts.
If the memory is not initialized, inspect the repository and record that state.
Preserve frozen evidence. Do not run experiments merely to catch up.
After each authorized milestone, update the project-memory records.
Current user instructions and applicable higher-priority instructions take precedence.
```

记录目录应放在仓库内且不被忽略，便于用户之后自行提交或导出。用 `git status --short`/`git check-ignore` 核验；若现有 ignore 规则阻止记录，不自动改旧规则，报告路径与原因。不要自动 git add、commit、push。最终明确：哪些记录只在本地、哪些内容尚未提交。

### M7. 新增验收条件

新增 no-training tests 检查 run-index JSON 可解析、路径可解析/缺失明确标记、status 与 scientific_status 区分、同一 run_id 的事件关联以及历史不覆盖；不得为测试记忆功能启动旧 suite 或真实训练。

最终交付必须包括：专用记录目录、更新后的 CATCH_UP、完整本次 session、决策与 run 索引、用户任务归档、根 AGENTS 入口的处理结果、git 状态，以及 frozen-file 核验。单独在最终聊天中总结但不写这些本地记录，视为任务未完成。

## 3. 阶段 A：先做数学与结构诊断，禁止神经网络训练

### A1. 实际对称与可辨识对象

严格分开五层：normalized density p；scalar g；coefficient field b；分解 beta/u/h；神经参数 theta。

先证明或反驳下列源码相关命题，并以合法 state 变换在网格/独立点验证：
- `h(0)=0` 和 `x(0)=0` 已固定 `g(0)=0`。`h→h+c, beta→beta-cu` 虽保持抽象 field，却除 c=0 外不属于当前 anchored h 类。不要把 centering h 当成必须补上的训练修复。
- 对连续域中完全相同的 density，g 只差常数；当前 anchor 加连续性固定该常数。有限网格相同不能自动推出连续域相同。
- 在归一化坐标 x 中，连续 h 且 h(0)=0 给出 `g(x)=x^T beta+o(||x||)`，故 `∇_x g(0)=beta`。验证推导；不要在有 anchor 的模型里默认 beta 可任意 trade off。
- `u→-u` 与 `cawnn.readout.weight→-weight` 是待验证的精确 sign symmetry，利用 tanh 的奇性。验证 parameter penalty 不变。
- `u→c*u, h→h/c` 在抽象分解层成立，但外层 tanh 不保证相同有限网络内任意 c 的精确实现。不要将 readout 除 c 当成精确实现。
- 验证至少一种 ASPNN 隐层 ReLU 正缩放/神经元置换；它保持网络函数但可改变参数表示。验证第一层 input weights 整体反号在 axial average 下保持 proximity 函数。不要把 θ-null directions 自动称为 field-null directions。

canonicalization 仅用于 evaluation：默认保留 reference gauge，不把 h 改成 mean-zero 后反馈训练；可用 `v=u/||u||, f=||u||*h` 加确定 sign convention 描述 anchored rank-one field。这个描述不代表原网络存在任意精确缩放变换。branch 接近零时 direction/angle/correlation 标记 undefined；不要同时强制 ||u||=1 与 RMS(h)=1，除非新增独立幅度且仅用于报告。

### A2. Teacher recipe 的特殊性

推导未加 .01 seed 扰动的 recipe：ASPNN 只依赖 perpendicular offset，CAWNN 初始通道使 raw 是12个 proximity 的平均。令
`a=(-1.2*sin(.35), .8*cos(.35))`，归一化坐标中应有未扰动 `h0(x)=H(a^T x)`，对称 anchors 使 H 为偶函数且 H(0)=0。

这使未扰动 h0 在整条 `a^T x=0` 直线上为零。分析
`q=(u^T x)*H(a^T x)`
在放大的连续 anchored rank-one 函数类中的另一个分解：
`u_alt=a`, `h_alt=(u^T x)*H(a^T x)/(a^T x)`，在零线上用正确的连续极限。
不得直接宣布 h_alt 属于当前固定 675 参数网络；必须明确这是候选结构机制，不是已证明的有限网络 counterexample。

针对正式冻结 teacher 做 forward-only 检查：在 `a^T x=0`、`u^T x=0` 上各取129个域内等距点；另以181个预定方向、每方向65个域内点描述 through-origin h 零线/近零线，包含 origin。记录归一化线 RMS、max、离原点半径，不根据结果改变 scan。

说明反证条件：若正式 teacher 的 h 在任何与 u 的零线不同的穿原点直线上都不恒为零，则同一 g 的另一个连续 anchored rank-one 表示不能任意旋转 u；这可能识别 field。有限点扫描不能证明全域条件。把未扰动的近一维构造、.01扰动是否打破该结构、实际网络可表示性严格分开。

Teacher preflight 曾因 u=(4,2) 的 q std=.09989648 未过 .1 而改为 (6,3)；仅记录事实，不撤销 teacher。指出 q std 门槛不等于“global-only 无法吸收的信号”门槛。

### A3. Penalty mismatch 的确定性检查

核验 teacher R_beta=.0013、R_u=4.5、R_theta≈.010511842925144，总 R≈4.511811842925144。

对匹配 teacher density 的 population 目标定义
`F_n = n*(logZ - E_pstar[g]) + R`, n_ref=256。
沿 `u(eps)=(1-eps)*u_star`，其余不动，数学上应有：
`dF_n/deps at 0 = -.2*||u_star||² = -9`；likelihood 一阶项为零，二阶项由 n*Var_pstar(q) 给出。
用 eps={1e-4,1e-5,1e-6} 做一维 forward/finite-difference 核验，不运行优化。
再分析同时缩小 beta/u 的可行方向：对非均匀 truth，正的 beta/u L2 惩罚使任何有限参数的 exact-density 表示都不能作为该 penalized population 目标的 stationary optimum。区别“确定存在 shrinkage bias”和“它是否足以解释 collapse”；后者必须量化，不预设结论。

记录 sum/mean 缩放：与旧 n=256 目标对应的是 `Lpop+R/256`，不是 `Lpop+R`。n44 与 n256 固定同一 SUM λ 时，per-event 有效 regularization 相差256/44，旧样本量对比并非固定正则强度的纯 sample-size 对比。

### A3b. 检查一个实际结构性的 collapsed stationary point

不要只把低梯度 collapse 解释成接近正确解。令 u=0、全部神经参数 theta=0、beta为同一目标的global-only最优值。核实当前网络此时 h=0 且空间参数的数据梯度为零，故这是完整目标的stationary point。

进一步审查局部展开：当前bias-free readout与零CAWNN使raw_spatial=O(||theta||²)，因而h=O(||theta||²)，q=O(||u||*||theta||²)。当三个L2系数均正时，branch penalty的正二次项支配数据项的三阶及更高阶变化；结合global beta块的严格凸性，这可构成一个strict collapsed local minimum。无惩罚时仍可能是非最优的一阶stationary point，而不是合法的最优density。

先给出有条件证明，核查所有假设，再用global-control beta的合法零branch state检查梯度和固定有限位移。至多3个seed20260919生成的单位方向，relative步长1e-2/1e-3/1e-4；不开展逃逸优化。这个数学点不等于旧collapsed checkpoint（旧theta可能仍很大），不能据此断言已定位每条历史collapse的实际机制。

### A4. Quadrature 与 KL identity

复用 `quadrature(r)` 的非等面积正权重，明确 density 值与 probability mass：
`pi_star_i = w_i*exp(g_star_i)/Z_star`，固定并 detach；
`Lpop_Q(g)=logsumexp(log(w)+g)-sum(pi_star*g)`。
验证
`Lpop_Q(g)-Lpop_Q(g_star)=KL(pi_star || pi_g)`。
必须同时输出 teacher-copy gap、gradient、penalty-only gradient，避免只复制 state 而未检查 objective。

同网格 zero gap 是离散一致性，不是连续积分精确的证明。读取原 P1-D LR checkpoint，在24/48/96/192中按下述预算诊断 logZ 与 KL，不能只引用旧 teacher refinement：
- teacher 与所有9个固定 step2000 state：48、96、192；
- 不重新 forward 全部18009历史状态，不选 truth-best checkpoint。
特别核验 .001/71 终点已有 `logZ48-logZ24≈.0453267748`、SUM目标修正约11.604这一事实。

## 4. 阶段 B：不训练网络，先求 global-only 和 fixed-h 凸对照

### B1. Global-only 是本阶段首要回答

当前矩形域和坐标 covariates 允许解析 normalizer：
`Z_global(beta)=9600*prod_j(sinh(beta_j)/beta_j)`，零点按极限处理。
实现稳定的 log-sinhc、均值函数 `coth(b)-1/b` 和导数，使用小 b 的 series 与大 |b| 的稳定表达式，不使用造成 overflow/cancellation 的裸公式。

Teacher 的 `E_pstar[x]`、`E_pstar[gstar]` 和 logZ 在48/96/192网格计算；global-only normalizer 本身用解析式。用两个单调一维 root solves 得到无惩罚人口最优 beta，不做 Adam 或参数搜索。加入 beta penalty 的对照同样可用单调求根。若不能达到残差/积分要求，报告 unresolved，不把可行拟合值当最优值。

输出：
- `delta_global = inf_beta KL(pstar || p_global_beta)`；这是 continuous integral 的受控数值近似，不把 Q48 值称为解析精确。
- 最优 beta、moment residual、收敛信息、44*delta、256*delta；后两者只是信息量尺度，不能变成 power/不可能性结论。
- 未惩罚 global optimum、带旧 beta ridge 的 global optimum、teacher、旧 .01/23 和 .01/71 collapsed states 的公平 comparison。
- 对 pure global representation 可把 u 和全部神经参数设零：其 R_theta=R_u=0；不要把旧 collapsed checkpoint 中与输出无关的残留 θ penalty 当成 global-only 最低代价。
- 比较 `256*KL+R-R_teacher`，同时展示 likelihood、各 penalty，不只 total。
- 面积测度和 teacher 测度下 q 对 {1,X} 的加权投影残差；报告 residual/std(q) 与 residual/std(g)。projection 与 KL optimum 不可互换。

已有 ZIP 中全部 step2000 surfaces 的只读重算给出参考值（仅48²离散值，不是验证答案）：.01/23 KL≈.00749231977；.01/71≈.00726647090；.001/71≈.13037927363。不得使用这些数值选样本或修改阈值。须独立复核，保留差异。

### B2. Teacher-h oracle 四参数对照

固定 h=h_star，仅估计 beta/u，设计列是 `[X, diag(h_star)X]`。这是明确标注 oracle 的凸诊断，不是普通 student training，也不是 θ 可恢复的证明。
在同一 population quadrature 上检查去除常数后的加权设计秩/condition number，并分别求：无惩罚、仅旧 beta/u penalty（θ penalty 是固定常数）。
固定初值 beta/u=(0,0)，用明确记录的 Newton/damped-Newton 凸求解，最多100迭代、Armijo(1e-4)、固定最大30次二分回溯。奇异设计先按已冻结 SVD cutoff 处理，不偷偷加 ridge。teacher 的 beta/u 是无惩罚问题的可行零-KL见证；不要求在秩不足时唯一恢复其参数。
不因为 oracle 容易恢复，就声称 full network 的优化或 field identification 已解决。

## 5. 阶段 C：Jacobian / local field-null diagnostics

只对以下3个冻结 state 先做：teacher、init1011、.01/23 step2000 collapsed state。不扫描全部训练历史。
主网格48²，96²验证弱方向；24²仅展示维度限制。明确：24² probability-mass map 的 rank最多575，而参数675，因此至少100个 local parameter-null directions 是网格维数限制，不能当成连续结构不可辨识的证据。

记录参数名字/形状/flatten顺序。使用可微 forward；旧 `rank_one_value_and_grad` 返回 detached gradients，不能直接拿来求 Hessian。
计算/核验：
`Jg=[X, diag(h)X, diag(Xu)Jh]`；
`Jb=d vec(beta+u*h)/d eta`；
`Jg_centered=Jg-1*(pi_star^T Jg)`；
`F=Jg_centered^T diag(pi_star) Jg_centered`（这是 teacher measure 下的函数敏感度；只有在匹配 truth 的点才直接对应未惩罚 population Hessian）；
`K=Jb^T diag(area_measure repeated per coefficient) Jb`。

区分：Jb v≈0 的表示方向；Jg_centered v≈0 但 Jb v明显非零的 field-invisible方向；两者非零但比例很小的弱方向。可以在 K 的正谱子空间内求 generalized spectrum Fv=lambda Kv，另存 raw spectra，不只给秩。
SVD relative cutoffs 固定为1e-6/1e-8/1e-10并列报告；主 generalized whitening cutoff=1e-10，记录敏感性。不能用单位不同的原始参数 singular value 直接比较“哪个参数更可辨识”。
最多选3个按预声明规则排序的 field-changing最弱方向，做双向有限位移，relative parameter norms为1e-4/1e-3/1e-2。记录实际 field变化、raw/centered g变化、KL、penalty，并在96²和固定独立点验证。Taylor-null 不等于 exact finite alternative，没找到方向也不证明 global identification。

## 6. 阶段 D：仅在基础检查通过后执行最小 population trajectories

若模型/target/梯度/积分检查失败，停止于 blocked 或 numerical_unresolved，不运行 full-network population，更不能进入 finite-sample。
否则从归档 init1011 运行恰好两条主轨迹：
P0：`F=256*Lpop_Q`，所有 penalty=0。
Pλ：`F=256*Lpop_Q+R`，R沿用旧数值。

两条只改变 penalty，其他完全相同：原模型、float64、CPU1线程、Adam(lr=.003, betas=(.9,.999), eps=1e-8, weight_decay=0, amsgrad=False)、2000更新、训练48²。选择.003只是固定已有诊断条件，不宣称它最优。不要混入 L-BFGS、scheduler、clipping、restarts 或不同 initial seeds。

逐步保存 data objective、total、各 penalty、block-gradient L2、真实 block-update L2；每25步记录 h/q 和固定 recovery；checkpoints沿用0/100/300/600/1000/1500/2000，保存 model+Adam state。96²作独立监测；192²只评价 teacher、固定终点及基础凸对照。训练积分始终48²，不在运行中换网格。

额外记录 raw_spatial/tanh 的范围和饱和比例、ASPNN/CAWNN ReLU active fractions、reference branch 与 location branch 对梯度的贡献、||u||、h RMS/std、q RMS/std，帮助区分 dead/constant/saturated branch。诊断不能改变 backward 或 optimizer。

出现非有限数立即保存失败现场并停止该轨迹；若独立积分明显不一致，保留完整结果但标记 grid-unresolved，禁止将失败归因于 identification。不得选一个更好 checkpoint 替代规定终点。

仅当 P0 cold-start 未达到下面 density success 且积分/实现检查通过，允许一条 oracle-local P0：teacher trainable parameter vector 加固定 seed20260918 的 Gaussian 单位方向扰动，扰动范数为 `1e-3*max(1,||eta_star||)`，buffers不变；其余训练配置完全同P0。这只检查 local recoverability，不计入正常 cold-start成功率，不用其参数 warm-start 其他实验。最多这一条，不搜索扰动。必须报告 oracle 初始KL：它可能在 step0 已满足绝对density门槛，因此“末步通过”本身不能证明恢复。另要求观察初始excess KL是否降低至少90%且末段稳定；若初始gap已低于10倍数值floor，则标记该oracle只检验局部稳定性，不能检验有分辨力的局部恢复；不要为此事后放大扰动重跑。

如 static/global controls 已回答某个问题，仍不得把 population gate改成预期结论。报告哪些问题已确定、哪些两条轨迹仍无法区分。

## 7. 执行前固定 numerical / recovery / stopping 规则

将下面数值原样写入 protocol；它们是本轮操作定义，不是结构可辨识性定理或通用科学标准。

- state copy：g/field/p最大绝对误差≤1e-12；同网格 KL identity atol=rtol=1e-10；teacher未惩罚 per-event gradient max abs≤1e-9。
- 光滑方向 finite difference：记录三个epsilon，误差≤1e-7+1e-4*|analytic derivative|；ReLU kink单独标记，不能伪造二阶光滑性。
- global/fixed-h 凸求解：per-event moment/gradient infinity norm≤1e-10；达不到则不声称找到 optimum。
- numerical fidelity：96→192 的 |ΔlogZ|≤1e-4，|ΔKL|≤1e-6+.01*max(KL96,KL192)，|ΔTV|≤1e-3；global gap 按相同 KL 规则。失败标记 unresolved，不自动放宽或额外重训。
- numerical function-equivalence见证：独立网格上 KL≤1e-10、常数对齐后 max|Δg|≤1e-6；还需原点附近/独立点检查。只能称“在已测试节点和容差下”，不能当成连续域证明。
- P0 practical density success：192² KL≤1e-4 且 TV≤.01，且通过积分规则。另报 centered g 的面积/teacher加权 RMSE；predictor success定义为两者≤.02，不能与density success混成一个指标。
- 若已解析并数值确认的 global gap>1e-5，另报“空间信号解释率”是否达到 `KL_full≤.1*delta_global`。global gap不可靠时不计算这个比率。
- coefficient-field RMSE、relative RMSE、逐分量图独立报告；未建立 identification 前不得设其为所有训练的硬成功条件。v/f canonical decomposition只在非退化field上比较；θ-distance仅作内部描述。
- field贡献的面积RMS≤1e-12时，方向/相关性标记undefined；保留旧 h_std≤1e-12 和near≤1e-4标志供历史对照，但绝不以 branch非零作为success。
- 沿用旧末200步 grad L2/256≤1e-3、total range/256≤1e-5作为 descriptive stationarity flag；仍固定跑2000，不早停；该 flag不是 global optimum证明。
- Pλ 不以“exact teacher density/parameters”为通过条件。比较其 dense penalized objective 与 teacher、pure-global、fixed-h oracle这些已知可行值，同时报告未惩罚KL。若高于已知可行点，说明当前求解未达到更好已知目标，不等于数学上不存在更好的解。

## 8. 解释与下一步决策，不把预期结论写死

必须分别讨论：
1. population p/g恢复，但 b或canonical decomposition不恢复：是否有真正的函数等价见证？还是小KL对低概率区域不敏感？还是仅内部网络表示不同？
2. 当前 anchored模型的完整连续h≡0若声称exact teacher density：先检验逻辑冲突。teacher q非零，且∇g(0)=beta；不能仅凭高相关称exact collapse成功。网格collapse也不等于连续域h≡0。
3. P0成功、Pλ改变density：量化penalty bias与目标改善，不能称Pλ是优化bug。
4. P0失败、oracle-local成功：支持cold-start/basin/conditioning困难，不证明全局不可识别。
5. P0与oracle-local都失败：先查gradient/积分/activation，然后才讨论更复杂optimizer，不先扩大sweep。
6. density正确但θ不同：通常不能作为科学失败；finite data恢复不好也不能直接由structural ambiguity解释。
7. 原n44/n256比较包含regularization scaling差异；原A/B只是两条切片，不是可分解数据/初始化方差的全交叉设计。

默认在此结束，不启动新finite-sample训练。用 `NEXT_DECISION.md` 给出下一轮最小 matched comparison：只有在population基线可信后，才建议固定归档data23、init1011、同grid/optimizer/penalty，对比population与empirical；是否再加入n44 prefix和第二个seed必须有问题驱动。不能把旧24² empirical run直接与新48² population run的差异全归因于finite sample。

## 9. 验收输出

新增独立 tests，只运行新测试文件，覆盖数学/输入/targets detach/权重/analytic normalizer/penalty scaling/symmetry/canonical描述/梯度/Jacobian/输出拒绝覆盖/old-file integrity；test不得暗中启动2000步训练或旧完整suite。

最后报告：实际读取的snapshot；旧事实核验与必要纠正；明确成立的数学结论；全部静态诊断与凸controls；已运行轨迹总数及每条完整结果；未执行项及原因；五层recovery分开的结论；尚不能区分的机制；下一步只推荐一个最小实验。不要把未达到事先门槛的结果写成“基本通过”，也不要为了整齐删除或重跑失败case。


另外必须完成第2.1节的项目本地记录验收。在最终回复中明确给出 `docs/project_memory/CATCH_UP.md` 的实际路径、最后更新时刻、对应代码状态和本次 session 路径；未经当前用户另行授权，不提交或推送这些文件。
