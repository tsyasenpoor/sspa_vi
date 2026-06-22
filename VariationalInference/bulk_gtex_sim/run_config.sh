#!/bin/bash
# Run ONE full bulk-sim config end-to-end: (reuse/Generate X0) -> truth -> inject -> labels ->
# emit -> DRGP fit -> recovery. X0 (SPsimSeq null) is cached per (sim_seed,n_sim,n_genes) and
# reused across effect/K/gamma configs (SPsimSeq is the slow step; the null is config-independent).
#
# Usage:
#   run_config.sh --n-sim 2000 --n-genes 18116 --sim-seed 0 \
#       --effect 3.0 --truth-seed 0 --k-fit 15 --rw 1.0 --gamma-prs 0.0 \
#       --out-root data/Simulations/bulk_gtex_v1
# NB: no `set -u` — conda's compiler-env (de)activation scripts reference unbound vars
# (CONDA_BACKUP_CXX) and abort under -u when switching bulksim_r <-> jax_gpu.
set -eo pipefail

# defaults
NSIM=2000; NGENES=0; SIMSEED=0; EFFECT=3.0; TSEED=0; KFIT=15; RW=1.0; GAMMA=0.0
OUTROOT=data/Simulations/bulk_gtex_v1
PRS=""                                   # optional .npy of z-scored PRS (len=n_sim)
while [ $# -gt 0 ]; do case "$1" in
  --n-sim) NSIM=$2; shift 2;; --n-genes) NGENES=$2; shift 2;; --sim-seed) SIMSEED=$2; shift 2;;
  --effect) EFFECT=$2; shift 2;; --truth-seed) TSEED=$2; shift 2;; --k-fit) KFIT=$2; shift 2;;
  --rw) RW=$2; shift 2;; --gamma-prs) GAMMA=$2; shift 2;; --out-root) OUTROOT=$2; shift 2;;
  --prs) PRS=$2; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done

ROOT=/labs/Aguiar/SSPA_BRAY
SCR=$ROOT/BRay/VariationalInference/bulk_gtex_sim
source /home/FCAM/tyasenpoor/miniconda3/etc/profile.d/conda.sh
RS=/home/FCAM/tyasenpoor/miniconda3/envs/bulksim_r/bin/Rscript
cd "$ROOT"
NGARG=""; [ "$NGENES" != "0" ] && NGARG="--n-genes $NGENES"

# 1. X0 null background (cached)
X0=$OUTROOT/X0/X0_seed${SIMSEED}_n${NSIM}_g${NGENES}.tsv.gz
if [ ! -s "$X0" ]; then
  echo "[X0] generating $X0"
  $RS $SCR/fit_spsimseq.R --ref $OUTROOT/wb_reference_counts.tsv.gz \
      --n-sim $NSIM $NGARG --seed $SIMSEED --out "$X0"
else echo "[X0] reuse $X0"; fi

# per-config dir
CFG=$OUTROOT/cfg_e${EFFECT}_t${TSEED}_k${KFIT}_rw${RW}_g${GAMMA}_s${SIMSEED}
mkdir -p "$CFG"

conda activate jax_gpu; export PYTHONPATH=$ROOT/BRay
# 2-3. truth + inject
python $SCR/make_truth.py --x0 "$X0" --out-dir "$CFG" --k 10 --n-disease 3 \
    --prog-size 100 --effect $EFFECT --truth-seed $TSEED
$RS $SCR/inject_thindiff.R --x0 "$X0" --beta "$CFG/beta_coef.tsv.gz" \
    --theta "$CFG/theta_design.tsv.gz" --out "$CFG/X_injected.tsv.gz" --seed $TSEED
# 4-5. labels + emit
PRSARG=""; [ -n "$PRS" ] && PRSARG="--prs $PRS"
python $SCR/make_labels.py --truth "$CFG/truth.npz" --out-dir "$CFG" \
    --prevalence 0.26 --gamma-prs $GAMMA $PRSARG
AUXARG=""; awk "BEGIN{exit !($GAMMA!=0)}" && AUXARG="--aux-columns PRS"
python $SCR/emit_dataset.py --injected "$CFG/X_injected.tsv.gz" --labels "$CFG/labels.npz" \
    --out "$CFG/simulated/ds.csv.gz"
# 6. DRGP fit
python -u $ROOT/BRay/VariationalInference/quick_reference.py --data "$CFG/simulated/ds.csv.gz" \
    --label-column heart_disease $AUXARG --mode unmasked --n-factors $KFIT --max-iter 3000 \
    --tol 0.001 --regression-weight $RW --early-stopping elbo --seed 0 --output-dir "$CFG/drgp_fit"
# 7. recovery
python $SCR/check_recovery.py --fit-dir "$CFG/drgp_fit" --truth "$CFG/truth.npz" \
    | tee "$CFG/recovery.txt"
echo "CONFIG DONE -> $CFG"
