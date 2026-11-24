#!/usr/bin/env bash
# Reproduce every number and figure of the ESANN 2027 paper (EXPOSURE), from scratch or from existing checkpoints.
#
#   scripts/reproduce_paper.sh all       train every agent, score, draw the figures, check the numbers (days on one GPU)
#   scripts/reproduce_paper.sh score     everything after training (needs the checkpoints in results/; hours)
#
# Single stages: train | score | figures | numbers. "numbers" regenerates Table 1 and checks every number of
# ../paper/main.tex against the result files (scripts/paper_numbers.py; exit code 1 on any mismatch).
# Setting: TD agents, width 256 (2 hidden layers, ReLU, otherwise the xrl Direct-Q configuration), 3 seeds, matched
# update counts. Table 1 baselines: MA-DQN and DQN + surrogate with variance and Q^2 heads (scripts/certainty_baselines.py,
# our re-implementation of the xrl recipe); the xrl agents still give the replay reference. Table 1 reports the mean over
# boards of the per-board Spearman rho and EXPOSURE's v_obs through the decoder's completions (paper_table.py defaults;
# --stat median / --vobs latent for the alternatives). Run detached if needed:
#   systemd-run --user --working-directory=$PWD bash scripts/reproduce_paper.sh all   (setsid/nohup jobs die with the session)
set -uo pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
L=results/logs/fix
C=results/logs/certainty_baselines
mkdir -p "$C"
P=results/philipp
mkdir -p "$L"
PAR=${PAR:-3}  # parallel jobs per stage (GPU memory: 3 fit on 32 GB)

RANDOM_U=(165290 165847 165839)  # updates matched to the xrl agents' budget, per seed
PM05_U=(186075 186720 186324)
MS_U=(157853 158532 157568)
D=0.9999379771968211  # perturbed-minimax annealing, as in xrl
XRL="--algo dqn --value-head scalar --gamma 1.0 --batch-size 512 --replay-capacity 100000 --replay-warmup 5000 \
--target-sync-interval 1000 --epsilon-start 1.0 --epsilon-end 0.05 --learning-rate 0.001 --no-plots --no-mlflow \
--hidden-sizes 256,256"

run_jobs() {  # run the commands in file $1, $PAR at a time; report the jobs whose log has a traceback
  xargs -P "$PAR" -I{} bash -c "{}" < "$1"
  grep -o '> [^ ]*\.txt' "$1" | cut -c3- | xargs grep -l Traceback 2>/dev/null | sed 's/^/  failed: /'
}

train() {
  local jobs; jobs=$(mktemp)
  for s in 0 1 2; do
    # EXPOSURE (ours)
    echo "uv run python scripts/train_td.py --xrl-matched --width 256 --var-weight 50 --seed $s --updates ${RANDOM_U[$s]} > $L/td_random_w256_s$s.txt 2>&1" >> "$jobs"
    echo "uv run python scripts/train_td.py --xrl-matched --width 256 --var-weight 50 --opponent pm05 --anneal-opponent --seed $s --updates ${PM05_U[$s]} > $L/td_pm05a_w256_s$s.txt 2>&1" >> "$jobs"
    echo "uv run python scripts/minesweeper.py --mines 3 --xrl-matched --width 256 --td-updates ${MS_U[$s]} --seed $s > $L/ms_td_w256_s$s.txt 2>&1" >> "$jobs"
    # baselines: mask-aware DQN (Direct-Q) and the standard DQN that the surrogate explains
    for kind in directq dqn; do
      flags=$([ "$kind" = directq ] && echo "--feature-mask-input --masked-batch-ratio 1" || echo "--masked-batch-ratio 0")
      echo "uv run python -m xrl.cli train --env tictactoe --opponent random $XRL $flags --episodes 48300 --epsilon-decay-episodes 23400 --seed $s --out $P/ttt_${kind}_scalar_w256/seed_$s > $L/xrl_ttt_${kind}_w256_s$s.txt 2>&1" >> "$jobs"
      echo "uv run python -m xrl.cli train --env tictactoe --opponent perturbed_minimax --perturbed-minimax-discount $D $XRL $flags --episodes 48300 --epsilon-decay-episodes 23400 --seed $s --out $P/ttt_pm05_${kind}_scalar_w256/seed_$s > $L/xrl_ttt_pm05_${kind}_w256_s$s.txt 2>&1" >> "$jobs"
      echo "uv run python -m xrl.cli train --env minesweeper --rows 4 --columns 4 --mines 3 $XRL $flags --episodes 32800 --epsilon-decay-episodes 15900 --seed $s --out $P/ms4x4_3_${kind}_scalar_w256/seed_$s > $L/xrl_ms_${kind}_w256_s$s.txt 2>&1" >> "$jobs"
    done
  done
  run_jobs "$jobs"
  # Minesweeper surrogate: xrl recipe (epsilon-greedy rollouts, eps 0.1); tic-tac-toe surrogates are trained in score
  : > "$jobs"
  for s in 0 1 2; do
    echo "uv run python scripts/train_scalar_surrogate.py --width 256 --base-run $P/ms4x4_3_dqn_scalar_w256/seed_$s --out $P/ms4x4_3_dqn_surrogate_scalar_w256/seed_$s --seed $s > $L/sur_ms_w256_s$s.txt 2>&1" >> "$jobs"
  done
  run_jobs "$jobs"
  # Table 1 baselines with certainty heads (train + score); width 512 and Q-only as checks (setup text)
  : > "$jobs"
  for s in 0 1 2; do
    for d in random pm05 ms; do
      for v in "ma" "surr" "ma --width 512" "ma --q-only"; do
        tag=$(echo "$v" | tr -d ' -')
        echo "uv run python scripts/certainty_baselines.py --domain $d --method $v --seed $s > $C/${d}_${tag}_s$s.txt 2>&1" >> "$jobs"
      done
    done
  done
  run_jobs "$jobs"
  echo "train done $(date)"
}

score() {
  local jobs; jobs=$(mktemp)
  for s in 0 1 2; do  # tic-tac-toe surrogates on the DQN's exact epsilon-greedy occupancy (trains them if missing)
    echo "uv run python scripts/score_surrogate_occupancy.py --opponent random --seed $s > $L/surrogate_eps10_random_s$s.txt 2>&1" >> "$jobs"
    echo "uv run python scripts/score_surrogate_occupancy.py --opponent pm05 --seed $s > $L/surrogate_eps10_pm05_s$s.txt 2>&1" >> "$jobs"
  done
  run_jobs "$jobs"
  # Table 1: mean game and v_mask (own + replay matching) for every agent; v_state; rank columns; Shannon H
  : > "$jobs"
  echo "uv run python scripts/replay_matching_eval.py --all-td --xrl-matched --width 256 --replay 10000 --domain tictactoe --opponent random > $L/eval_replay_ttt_random_w256.txt 2>&1" >> "$jobs"
  echo "uv run python scripts/replay_matching_eval.py --all-td --xrl-matched --width 256 --replay 10000 --domain tictactoe --opponent pm05 > $L/eval_replay_ttt_pm05_w256.txt 2>&1" >> "$jobs"
  echo "uv run python scripts/replay_matching_eval.py --all-td --xrl-matched --width 256 --replay 10000 --domain minesweeper > $L/eval_replay_ms_w256.txt 2>&1" >> "$jobs"
  for s in 0 1 2; do
    echo "uv run python scripts/baselines.py --all-td --xrl-matched --width 256 --domain tictactoe --opponent random --seed $s > $L/baselines_ttt_random_w256_s$s.txt 2>&1" >> "$jobs"
    echo "uv run python scripts/baselines.py --all-td --xrl-matched --width 256 --domain tictactoe --opponent pm05 --seed $s > $L/baselines_ttt_pm05_w256_s$s.txt 2>&1" >> "$jobs"
  done
  echo "uv run python scripts/score_rank_columns.py --width 256 > $L/rank_columns_w256.txt 2>&1" >> "$jobs"
  echo "uv run python scripts/predictive_entropy.py --domain tictactoe --k 32 --r 4 > $L/predictive_entropy_tictactoe.txt 2>&1" >> "$jobs"
  echo "uv run python scripts/predictive_entropy.py --domain pm05 --k 32 --r 4 > $L/predictive_entropy_pm05.txt 2>&1" >> "$jobs"
  echo "uv run python scripts/predictive_entropy.py --domain minesweeper --k 16 --r 4 > $L/predictive_entropy_minesweeper.txt 2>&1" >> "$jobs"
  # text: own-policy v_state reference; chain rule, parity check, probes, optimal win rates
  for d in tictactoe pm05 minesweeper; do
    for s in 0 1 2; do
      echo "uv run python scripts/score_own_state.py --domain $d --width 256 --seed $s > $L/own_state_${d}_s$s.txt 2>&1" >> "$jobs"
    done
  done
  for section in chain_rule parity probes optimal; do
    echo "uv run python scripts/paper_analyses.py $section > $L/paper_analyses_$section.txt 2>&1" >> "$jobs"
  done
  run_jobs "$jobs"
  # Table 1 baselines rescored from their checkpoints; EXPOSURE's v_obs through the decoder's completions
  : > "$jobs"
  for s in 0 1 2; do
    for d in random pm05 ms; do
      for v in "ma" "surr" "ma --width 512" "ma --q-only"; do
        tag=$(echo "$v" | tr -d ' -')
        echo "uv run python scripts/certainty_baselines.py --domain $d --method $v --seed $s --reuse > $C/rescore_${d}_${tag}_s$s.txt 2>&1" >> "$jobs"
      done
    done
  done
  run_jobs "$jobs"
  # Table 1 phi^inh: against each method's own full-observation variance (agent) and own greedy policy (own, text)
  : > "$jobs"
  for s in 0 1 2; do
    for d in random pm05 ms; do
      for m in exposure ma surr "ma --width 512"; do
        tag=$(echo "$m" | tr -d ' -')
        for ref in agent own; do
          echo "uv run python scripts/score_own_inherent.py --domain $d --method $m --seed $s --reference $ref > $L/own_inherent_${d}_${tag}_${ref}_s$s.txt 2>&1" >> "$jobs"
        done
      done
    done
  done
  run_jobs "$jobs"
  uv run python scripts/imputation_vobs.py --domains random pm05 ms > $L/imputation_vobs.txt 2>&1
  # text only (not Table 1): Monte Carlo noise of the model games, one Minesweeper checkpoint, three sampling repeats
  uv run python scripts/mc_noise.py > $L/mc_noise.txt 2>&1
  echo "score done $(date)"
}

figures() {
  # Figure 1: EXPOSURE's maps of the Figure 2 board (seed 2, board 1728), then the schematic around them
  uv run python scripts/architecture_maps.py --board 1728 --seed 2 > $L/figure_architecture_maps.txt 2>&1
  uv run python -c "import importlib.util as u; s = u.spec_from_file_location('a', 'scripts/architecture_figure.py'); m = u.module_from_spec(s); s.loader.exec_module(m); m.compact()"
  uv run python scripts/outcome_regions.py --checkpoint results/models/td_random_m10_var_xrl256_s2.msgpack --projection umap \
    --paper-out ../paper/figures/latent_outcomes.pdf > $L/figure_latent.txt 2>&1
  uv run python scripts/paper_figure_cells.py --width 256 --seed 2 --ttt-board 1728 > $L/figure_cells.txt 2>&1
  echo "figures done $(date)"
}

numbers() {
  uv run python scripts/paper_table.py  # defaults: --stat mean --vobs imputation
  uv run python scripts/paper_numbers.py
}

case "${1:-}" in
  all) train && score && figures && numbers ;;
  score) score && figures && numbers ;;
  train | figures | numbers) "$1" ;;
  *) sed -n '2,/^set -uo/p' "$0" | sed '$d'; exit 2 ;;
esac
