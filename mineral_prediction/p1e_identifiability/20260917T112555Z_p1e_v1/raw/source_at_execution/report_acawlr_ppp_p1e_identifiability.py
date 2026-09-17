"""Report P1-E strictly from saved artifacts, with no model imports or training."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def run(out):
    for name in ('SUMMARY.json','NOTES.md','MATH_REVIEW.md','SOURCE_AUDIT.md','NEXT_DECISION.md'):
        if (out/name).exists():raise FileExistsError(out/name)
    a=read(out/'STRUCTURE.json');b=read(out/'CONVEX_CONTROLS.json');c=read(out/'JACOBIAN.json');cl=read(out/'COLLAPSED.json');pop=read(out/'POPULATION_RESULTS.json');gate=read(out/'GATES.json');prov=read(out/'PROVENANCE.json')
    root=Path(__file__).resolve().parent
    locations={}
    for file,terms in {'acawlr_ppp_bridge.py':['class ReducedCAWNN','class ReducedRankOneSyntheticPredictor','def raw_spatial','def h(','class RankOnePenalty','def profiled_ppp'],
                       'acawlr_ppp_synthetic.py':['def quadrature'], 'acawlr_ppp_teacher_student.py':['def make_teacher','def state_hash']}.items():
        lines=(root/file).read_text(encoding='utf-8-sig').splitlines()
        for term in terms:locations[term]=f'{file}:{next(i+1 for i,s in enumerate(lines) if term in s)}'
    audit=['# SOURCE AUDIT','',f'Actual HEAD: {prov["HEAD"]}; branch {prov["branch"]}; implementation uncommitted.',
      '【源码事实】Actual repository has modules at root; AST source is ../scripts/ACAWLR_improved.py. Attachment prefixes are not current layout. Old files and signatures were not rewritten.',
      '【源码事实】x=xy/[60000,40000], b=beta+u*h, h=tanh(raw(xy))-tanh(raw(reference)), g=x^T b. No local intercept. '+locations['class ReducedRankOneSyntheticPredictor'],
      '【源码事实】ASPNN is 2→16→8→4→1, ReLU hidden/Sigmoid output; axial average uses offsets and negatives. ../scripts/ACAWLR_improved.py:ASPNN; '+locations['def raw_spatial'],
      '【源码事实】ReducedCAWNN has four channels, residual/channel/spatial attention, bias-free scalar readout, no BN/dropout. '+locations['class ReducedCAWNN'],
      '【源码事实】SUM PPP n*logZ-sum(g(events)); R=.01||beta||²+.1||u||²+.0005||theta||², all neural parameters included. '+locations['def profiled_ppp']+'; '+locations['class RankOnePenalty'],
      '【旧 artifact 事实】P1-C suite contains two failed recovery tests, not a fully passing suite: ACAWLR_PPP_P1C_TS_SUITE.json#/failures. No old tests rerun.',
      '【旧 artifact 事实】Rejected teacher u=(4,2), q std=.09989648397364517: ACAWLR_PPP_P1C_TS_TEACHER_PREFLIGHT_REJECTED.json#/spatial_score/std. Its replacement (6,3) preceded student fitting; teacher not changed here.',
      '【旧 artifact 事实】LR baseline 6d58b87c... differs from current HEAD; no reset performed. LR_PROTOCOL.json#/baseline_commit. Current environment is in PROVENANCE.json.',
      '【本轮数值诊断】Teacher/init loaded weights_only=True, CPU, complete state including buffers; hashes verified in PROVENANCE.json. No reconstructed random initial weights used.',
      '【本轮数值诊断】Teacher penalty and shrink direction: STRUCTURE.json#/penalty and #/shrink. All nine archived endpoints reevaluated without training: #/historical.',
      '【本轮数学推导】Anchoring, symmetries, candidate ridge decomposition and collapsed local minimum: MATH_REVIEW.md. Assumptions and limits retained.',
      '【尚未确定】Continuous-domain field identification, exact representability of the alternative h, and the mechanism of every historical collapsed checkpoint are not established by finite scans or local Jacobians.',
      '【本轮数值诊断】Frozen integrity is in FROZEN_VERIFY.json; no commit/push performed.']
    (out/'SOURCE_AUDIT.md').write_text('\n\n'.join(audit)+'\n',encoding='utf-8')
    mathtext='''# Mathematical review (conditions explicit)

1. **Five objects and the anchor.** On the connected rectangle, equality of strictly positive normalized densities implies g1-g2 is constant (almost-everywhere equality plus continuity suffices). Since x(0)=0, both g(0)=0; the constant vanishes. Since h is continuous and h(0)=0, x^T u h(x)=o(||x||); hence the normalized-coordinate derivative at zero is beta even without differentiability of h. Equality at finitely many nodes cannot substitute for continuous equality. The abstract shift h+c,beta-cu leaves b unchanged, but violates h(0)=0 unless c=0.

2. **Exact and inexact symmetries.** u→-u plus readout W→-W negates raw and h by tanh oddness, preserving b,g,p and all L2 terms. ReLU positive hidden-unit scaling with inverse outgoing scaling, and hidden-unit permutation, preserve the neural function; positive scaling can change its penalty. Negating all first-layer input weights swaps ASPNN(offset) with ASPNN(-offset) under axial averaging and preserves penalty. These are representation symmetries, not field-changing examples. Abstract u→cu,h→h/c is not generally implemented by dividing readout by c, because tanh(z/c)≠tanh(z)/c. Numerical comparisons use both independent grid and fixed points; no continuous proof follows from small numerical discrepancies alone. Evaluation canonicalization preserves anchoring, takes v=u/||u||, f=||u||h and deterministic sign; it is undefined for a zero field contribution.

3. **Recipe mechanism.** Before the fixed 0.01 perturbation, ASPNN(offset)=sigmoid(2|offset_perp|-2). Four positive center-copy channels, zero residual/attention logits and unit readout reduce raw to the mean of the 12 proximities (two sigmoid gates each contribute 1/2). Thus h0=H(a^T x), a=(-1.2 sin(.35), .8 cos(.35)), with H even and H(0)=0 by paired anchors. The finite set of projected anchors has no zero projection for this angle: near zero the recipe is smooth and H(t)=O(t²), stronger than mere continuity/evenness. Therefore H(t)/t→0, and h_alt=(u^T x)H(a^T x)/(a^T x), extended by zero on a^T x=0, is continuous and anchored. With u_alt=a it gives the same q in an enlarged continuous rank-one class. This is NOT proof that h_alt belongs to the fixed 675-parameter network. Mere continuous even H alone would not ensure this limit. The formal perturbed teacher is a separate object; finite line scans do not establish or exclude exact zero lines over the domain.

If another continuous anchored representation has identical g, beta is the same by the origin derivative. On u_alt^T x=0, equality requires (u^T x)h(x)=0. If u_alt is not parallel to u, h must vanish on that entire through-origin line by continuity. Absence of every such nonparallel zero line would force parallel u and hence the same field wherever the representation is defined. This is a conditional argument, not a conclusion from 181 directions.

4. **Penalty mismatch.** At a teacher density on the SAME quadrature, d Lpop=0 for every parameter direction. Along u(eps)=(1-eps)u*, dF/deps=-.2||u*||²=-9; likelihood curvature is 256 Var_p*(q). Jointly scaling beta and u by 1-eps scales g; at any nonuniform exact-density representation at least one is nonzero, so its beta/u L2 decreases to first order. Such an exact-density point cannot be stationary for positive beta/u penalty. This proves shrinkage bias, not that it explains historical collapse. SUM scaling corresponds to Lpop+R/256, and fixed SUM lambda gives 256/44 times stronger per-event penalty at n44 than n256. The historical A/B slices are not a crossed variance decomposition.

5. **A collapsed strict local minimum exists conditionally.** Set u and every neural parameter to zero, and beta to the global optimum of the SAME (quadrature or continuous) target, with matching beta penalty. ASPNN is 1/2, CAWNN feature activations are O(||theta||); bias-free readout implies raw=O(||theta||²), so h=O(||theta||²) uniformly on the compact domain and q=O(||u|| ||theta||²). Branch derivatives vanish at zero. For positive penalties on u and every theta coordinate, their quadratic increase dominates the cubic likelihood perturbation. The beta-only objective has positive definite covariance Hessian (and ridge); cross perturbations are higher order. Hence this is a strict local minimum in full parameters. ReLU need not have a classical Hessian at zero for this bound; do not infer smooth second derivatives there. Without penalties the point is stationary but this positive-quadratic proof fails. Historical collapsed states generally retain nonzero theta; the proof does not identify their actual basin/mechanism. COLLAPSED.json uses a separate discrete-global solution to avoid confusing analytic continuous beta with a discrete stationary point.

6. **Quadrature and convexity.** pi_i=w_i exp(g_i)/Z is probability mass, not the density value. Lpop_Q(g)-Lpop_Q(g*)=KL(pi*||pi_g) on the same positive-weight grid, with detached pi*. For rectangle area 9600, global Z=9600 prod(sinh(beta_j)/beta_j); derivatives are coth(beta)-1/beta and 1/beta²-csch²(beta), continued analytically at zero. Strict monotonicity gives root solves. Teacher moments still require numerical integration. Fixed-h design [X,diag(h*)X] is a four-parameter convex oracle, not ordinary student learning. Newton convergence is reported separately from fidelity. All controls carry likelihood and penalty separately.

7. **Jacobian scope.** Jg=[X,diag(h)X,diag(Xu)Jh]; Jb includes beta identity and h*u derivatives. Center Jg under teacher mass. F=Jgc^T diag(pi*)Jgc is a teacher-weighted function sensitivity, the unpenalized population Hessian only at matching truth. K=Jb^T diag(area repeated by coefficient)Jb measures field sensitivity. SVD cutoffs refer to weighted Jacobians; generalized whitening is on the positive singular subspace of weighted Jb (cutoff 1e-10 relative singular value, K eigenvalue ratio 1e-20). Sensitivity at 1e-6 and 1e-8 is retained. This parameterization-dependent numerical rank is not a global theorem. At 24², probability rank≤575 means at least 100 local parameter-null dimensions merely by grid size. Directions with Jb≈0 are representation-null; small centered Jg and nonzero Jb can be locally field-invisible or weak. Finite displacements are separately evaluated, not called exact alternatives.
'''
    (out/'MATH_REVIEW.md').write_text(mathtext,encoding='utf-8')
    summary=dict(run_id=out.name,HEAD=prov['HEAD'],execution='static_completed',scientific_status='passed' if gate['passed'] else 'numerical_unresolved',
        population=pop,failed_gates=gate['failed'],teacher_penalty=a['penalty'],global192=b['192']['global'],fixed_h_ridge192=b['192']['fixed_h_ridge'],
        historical_fidelity={k:v['fidelity'] for k,v in a['historical'].items()},frozen=read(out/'FROZEN_VERIFY.json')['passed'])
    write(out/'SUMMARY.json',summary)
    lines=['# P1-E results', '',f'Run: {out.name}; HEAD {prov["HEAD"]}. Static diagnostics completed; this does not imply all scientific gates passed.',
      f'Population trajectories executed: {pop["count"]}; status {pop["status"]}. Failed gates: {gate["failed"]}.',
      '', '| Control, 192² teacher moments | KL | TV | 256 KL + R - R_teacher | Solver |', '|---|---:|---:|---:|---|']
    for name in ('global','global_ridge','fixed_h','fixed_h_ridge'):
        v=b['192'][name];lines.append(f'| {name} | {v["KL"]:.10g} | {v["TV"]:.8g} | {v["penalized_minus_teacher"]:.9g} | {v.get("status",v.get("converged"))} |')
    lines+=['','Teacher moments and global gap are quadrature approximations; the global normalizer alone is analytic. The unpenalized controls still display their cost under the old penalty for fair feasible-point comparison.',
            '', '| Archived step2000 | KL48 | KL96 | KL192 | ΔlogZ96→192 | fidelity |','|---|---:|---:|---:|---:|---|']
    for k,v in a['historical'].items():lines.append(f'| {k} | {v["48"]["KL"]:.9g} | {v["96"]["KL"]:.9g} | {v["192"]["KL"]:.9g} | {v["fidelity"]["errors"]["logZ"]:.9g} | {v["fidelity"]["passed"]} |')
    lines+=['',f'0.001/71 24→48 check: {a["historical"]["0.001_71"]["24_to_48"]}.',
      '', 'Five-level interpretation: p/g equivalence under tested symmetries is observed; small KL is not a proof of global continuous equality. b identifiability remains conditional; no finite-network field-changing exact counterexample has been established. beta is fixed by the continuous origin derivative. u/h has sign/representation ambiguity; evaluation canonicalization does not modify training. theta has exact function-preserving transformations and need not match teacher weights.',
      '', 'The penalty directional derivative is negative at exact teacher density; positive ridge creates bias. A strict zero-branch local minimum exists under the conditions in MATH_REVIEW, but its relationship to the historical checkpoints remains unresolved. High correlation at a collapsed endpoint does not establish exact teacher density.',
      '', 'No new finite-sample training, real-data runs, optimizer search, teacher replacement, checkpoint selection, commit or push. Failure of archived endpoint integration is not evidence of structural nonidentifiability. The conservative predeclared gate blocks population when any required numerical fidelity check fails.',
      '', 'Consequently cases involving P0 success/P_lambda bias or oracle-local rescue cannot be judged if their trajectories were blocked. Existing finite-sample failures cannot be attributed uniquely to regularization, sampling, integration or basin effects.']
    (out/'NOTES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    nexttext=('# Next decision\n\nPopulation baseline is not yet validated. The smallest next action is a protocol review of whether failures confined to archived empirical endpoints should block fresh population controls, with the actual GATES.json evidence. Do not change this frozen run or launch a sweep. Only after an explicitly revised protocol and trustworthy population baseline, propose one matched population-versus-empirical comparison on archived data23/init1011, same grid/optimizer/penalty; do not compare old 24² empirical training against new 48² population and call all differences finite-sample effects.\n' if not gate['passed'] else '# Next decision\n\nPropose one matched population/empirical comparison: archived data23/init1011, same grid/optimizer/penalty. Do not execute it in this stage.\n')
    (out/'NEXT_DECISION.md').write_text(nexttext,encoding='utf-8')
    # Plots only consume raw arrays; never instantiate a model.
    cases=['teacher','0.01_23','0.01_71','0.001_71'];grid=torch.load(out/'raw/grid_96.pt',weights_only=True,map_location='cpu');surfaces=[grid['teacher']]+[torch.load(out/f'raw/legacy_{k}_96.pt',weights_only=True,map_location='cpu') for k in cases[1:]]
    fig,axes=plt.subplots(3,4,figsize=(13,9),layout='constrained')
    for i,(key,component) in enumerate([('g',None),('b',0),('b',1)]):
        data=[s[key] if component is None else s[key][:,component] for s in surfaces];bound=max(float(d.abs().max()) for d in data)
        for j,d in enumerate(data):
            im=axes[i,j].imshow(d.numpy().reshape(96,96).T,origin='lower',extent=(-60,60,-40,40),cmap='coolwarm',vmin=-bound,vmax=bound,aspect='equal')
            axes[i,j].set_title(cases[j]+': '+('g' if component is None else f'b{component+1}'),fontsize=10)
        fig.colorbar(im,ax=list(axes[i]),shrink=.7)
    fig.suptitle('Fixed states, 96²; same color scale within each field (nonuniform grid index image)');fig.savefig(out/'figures/fixed_fields.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for name in ('teacher','init1011','collapsed'):
        raw=torch.load(out/f'raw/jacobian_{name}.pt',weights_only=True,map_location='cpu');sv=raw['g_singular'].numpy();ax.semilogy(np.arange(len(sv)),np.maximum(sv,1e-30),label=name)
    ax.set(xlabel='Singular-value index',ylabel='Weighted centered Jg singular value',title='Local sensitivity spectra; not directly comparable parameter identifiability');ax.legend();ax.grid(alpha=.2);fig.savefig(out/'figures/spectra.png',dpi=140);plt.close(fig)
    write(out/'REPORT_PROVENANCE.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json') if p.name not in ('REPORT_PROVENANCE.json','SUMMARY.json')}))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True,type=Path);run(p.parse_args().input_dir.resolve())
