"""Entry point: python -m VariationalInference.Simulations.cli <subcmd> [args]."""
from __future__ import annotations
import argparse
import sys
from . import config


def _make_truths(args):
    from .truths import save_truth
    for t in range(args.G_truth):
        p = save_truth(t)
        print(f"  truth {t}: {p}")


def _compute_ss_type(args):
    from .calibrate import compute_ss_type
    val = compute_ss_type()
    print(f"  ss_type = {val:.4e}  -> {config.SIM_ROOT / 'ss_type.json'}")


def _calibrate(args):
    from .calibrate import run_calibration
    run_calibration(G_truth=args.G_truth)


def _generate(args):
    from .dataset import write_dataset
    p = write_dataset(args.truth_idx, args.h2, args.r, args.inner_seed,
                      is_stability=args.is_stability)
    print(f"  wrote {p}")


def _run_drgp(args):
    from .runner_drgp import run
    run(args.h5ad, args.mode, args.K, args.inner_seed, args.out_dir,
        regression_weight=args.regression_weight)


def _run_unsup(args):
    from .runner_unsup import run
    run(args.h5ad, args.method, args.K, args.inner_seed, args.out_dir)


def _run_gene(args):
    from .runner_gene import run
    run(args.h5ad, args.inner_seed, args.out_dir)


def _evaluate(args):
    from .evaluate import run
    print(run(args.result_dir, args.h5ad))


def _aggregate(args):
    from .aggregate import collect, bottleneck_join
    df = collect()
    print(f"  collected {len(df)} rows -> {config.SIM_ROOT / 'metrics.parquet'}")
    try:
        bn = bottleneck_join()
        print(f"  bottleneck rows: {len(bn)}")
    except Exception as e:
        print(f"  bottleneck skipped (no matched-cosine rows yet?): {e}")


def _stability(args):
    from .aggregate import stability_summary
    method_modes = [
        ("drgp", "unmasked"), ("drgp", "masked"),
        ("drgp", "pathway_init"), ("drgp", "combined"),
        ("nmf",  "lr"), ("schpf", "lr"), ("spectra", "lr"),
    ]
    df = stability_summary(method_modes)
    print(df.to_string(index=False))


def _plots(args):
    from .plots import run_all
    run_all()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="VariationalInference.Simulations.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_mt = sub.add_parser("make-truths", help="Generate G_truth structural truths")
    p_mt.add_argument("--G-truth", type=int, default=config.G_TRUTH)
    p_mt.set_defaults(func=_make_truths)

    p_st = sub.add_parser("compute-ss-type")
    p_st.set_defaults(func=_compute_ss_type)

    p_cal = sub.add_parser("calibrate")
    p_cal.add_argument("--G-truth", type=int, default=config.G_TRUTH)
    p_cal.set_defaults(func=_calibrate)

    p_g = sub.add_parser("generate")
    p_g.add_argument("--truth-idx", type=int, required=True)
    p_g.add_argument("--h2", type=float, required=True)
    p_g.add_argument("--r", type=float, required=True)
    p_g.add_argument("--inner-seed", type=int, default=0)
    p_g.add_argument("--is-stability", action="store_true")
    p_g.set_defaults(func=_generate)

    p_d = sub.add_parser("run-drgp")
    p_d.add_argument("--h5ad", required=True)
    p_d.add_argument("--mode", choices=["unmasked","masked","pathway_init","combined"], required=True)
    p_d.add_argument("--K", type=int, required=True)
    p_d.add_argument("--inner-seed", type=int, default=0)
    p_d.add_argument("--out-dir", required=True)
    p_d.add_argument("--regression-weight", type=float, default=None)
    p_d.set_defaults(func=_run_drgp)

    p_u = sub.add_parser("run-unsup")
    p_u.add_argument("--h5ad", required=True)
    p_u.add_argument("--method", choices=["nmf","schpf","spectra"], required=True)
    p_u.add_argument("--K", type=int, required=True)
    p_u.add_argument("--inner-seed", type=int, default=0)
    p_u.add_argument("--out-dir", required=True)
    p_u.set_defaults(func=_run_unsup)

    p_gn = sub.add_parser("run-gene")
    p_gn.add_argument("--h5ad", required=True)
    p_gn.add_argument("--inner-seed", type=int, default=0)
    p_gn.add_argument("--out-dir", required=True)
    p_gn.set_defaults(func=_run_gene)

    p_e = sub.add_parser("evaluate")
    p_e.add_argument("--result-dir", required=True)
    p_e.add_argument("--h5ad", required=True)
    p_e.set_defaults(func=_evaluate)

    p_a = sub.add_parser("aggregate")
    p_a.set_defaults(func=_aggregate)

    p_s = sub.add_parser("evaluate-stability")
    p_s.set_defaults(func=_stability)

    p_p = sub.add_parser("plots")
    p_p.set_defaults(func=_plots)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
