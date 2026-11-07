---
name: follower-test
description: Run the ego-plan follower test on a v2 checkpoint and compare it with the reference model
argument-hint: [checkpoint-path] [label]
disable-model-invocation: true
---

Run the follower test on checkpoint `$0` and label the results `$1`.

Purpose: check whether the world model reacts to the ego plan. A car following the ego
should be predicted to slow down when the ego brakes and to go further when the ego accelerates.

Steps:
1. Find the existing follower-test code in this repo (search for "follower") and reuse it.
   Do not write a new implementation and do not change model code.
2. Use the same 20 val scenes as before (an agent directly behind the ego in the same lane,
   within 15 m). Val only, never test.
3. For checkpoint `$0`, predict the follower under four ego plans: real, brake to stop,
   accelerate, and plan off.
4. Save results to `outputs/$1/follower_test.json` and a 2-example plot to
   `outputs/$1/follower_test.png`.
5. Report one table with these columns, with the reference model `checkpoints/v2_d_rerun`
   (best epoch) next to `$1`: ego plan, ego travel at 6 s, follower's predicted travel,
   and shift vs the real-plan prediction.
6. Give a verdict in one line each:
   - Reacts in the right direction? (brake → less travel, accelerate → more)
   - Stronger or weaker than the reference?
   - Physically plausible? (with brake to stop, does the follower stay behind the ego?)

Keep commands short, with a timeout. Do not commit anything.
