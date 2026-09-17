# SOURCE AUDIT



Actual HEAD: af633edd655a6e14398f59a222b69d07a347620b; branch codex-refactor; implementation uncommitted.

【源码事实】Correction: actual Git root is D:/code/ResearchPractice; this report uses paths relative to its mineral_prediction working directory unless prefixed otherwise. AST source is ../scripts/ACAWLR_improved.py. Attachment mineral_prediction/ prefixes are consistent with Git root. Earlier session called the module directory repository root incorrectly; old evidence was not rewritten.

【源码事实】x=xy/[60000,40000], b=beta+u*h, h=tanh(raw(xy))-tanh(raw(reference)), g=x^T b. No local intercept. acawlr_ppp_bridge.py:191

【源码事实】ASPNN is 2→16→8→4→1, ReLU hidden/Sigmoid output; axial average uses offsets and negatives. ../scripts/ACAWLR_improved.py:ASPNN; acawlr_ppp_bridge.py:224

【源码事实】ReducedCAWNN has four channels, residual/channel/spatial attention, bias-free scalar readout, no BN/dropout. acawlr_ppp_bridge.py:167

【源码事实】SUM PPP n*logZ-sum(g(events)); R=.01||beta||²+.1||u||²+.0005||theta||², all neural parameters included. acawlr_ppp_bridge.py:109; acawlr_ppp_bridge.py:250

【旧 artifact 事实】P1-C suite contains two failed recovery tests, not a fully passing suite: ACAWLR_PPP_P1C_TS_SUITE.json#/failures. No old tests rerun.

【旧 artifact 事实】Rejected teacher u=(4,2), q std=.09989648397364517: ACAWLR_PPP_P1C_TS_TEACHER_PREFLIGHT_REJECTED.json#/spatial_score/std. Its replacement (6,3) preceded student fitting; teacher not changed here.

【旧 artifact 事实】LR baseline 6d58b87c... differs from current HEAD; no reset performed. LR_PROTOCOL.json#/baseline_commit. Current environment is in PROVENANCE.json.

【本轮数值诊断】Teacher/init loaded weights_only=True, CPU, complete state including buffers; hashes verified in PROVENANCE.json. No reconstructed random initial weights used.

【本轮数值诊断】Teacher penalty and shrink direction: STRUCTURE.json#/penalty and #/shrink. All nine archived endpoints reevaluated without training: #/historical.

【本轮数学推导】Anchoring, symmetries, candidate ridge decomposition and collapsed local minimum: MATH_REVIEW.md. Assumptions and limits retained.

【尚未确定】Continuous-domain field identification, exact representability of the alternative h, and the mechanism of every historical collapsed checkpoint are not established by finite scans or local Jacobians.

【本轮数值诊断】Frozen integrity is in FROZEN_VERIFY.json; no commit/push performed.
