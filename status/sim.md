# status/sim.md (sim agent, v2-transformer)

Updated 2026-11-07.

## Done
- Final closed-loop test (test150: replay v4 + pre-registered reactive run). Decision: WM is safer on the
  close-call co-primary endpoint; at-fault collisions show no detectable difference (1 event).
  Report: `outputs/sim/test150_final_report.md`.
- pod-runs merged into v2-transformer (08ec065); 3-seed val minADE6 1.239 ± 0.006 m recorded in docs/EXPERIMENTS.md.
- README.md: summary, results, limitations, reproduce commands; dry run on a fresh clone with mini scenes only
  (fix in 40cf943). Branch pushed.

## Running
- Nothing.

## Needs
- Open items: reactive stage 2 (swept ego footprint, merge anticipation), weak WM follower reaction,
  coverage beyond 16 slots (see docs/EXPERIMENTS.md "Next steps").
