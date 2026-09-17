# 给现有 P1-E Codex 任务追加的本地记录要求

将以下部分纳入原任务：仅增加持久化记录与交接，不改变原实验阈值、预算、seed、frozen 边界和执行授权。

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

