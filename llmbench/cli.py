"""Command line interface: `llmbench run ...` and `llmbench show <result.json>`."""

from __future__ import annotations

import argparse
import json
import sys

from .runner import PRECISIONS, format_run, run_benchmark


def build_parser():
    p = argparse.ArgumentParser(prog="llmbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark sweep")
    r.add_argument("--model", required=True, help="HF model id or local path")
    r.add_argument("--precision", nargs="+", default=["fp16"], choices=PRECISIONS)
    r.add_argument("--batch", nargs="+", type=int, default=[1])
    r.add_argument("--prompt-len", nargs="+", type=int, default=[128])
    r.add_argument("--new-tokens", type=int, default=128)
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="results")

    s = sub.add_parser("show", help="print a saved result file as a table")
    s.add_argument("path")
    return p


def cmd_show(path):
    doc = json.load(open(path))
    env = doc["env"]
    print(f"{doc['model']} | {env.get('gpu') or doc['device']} | torch {env.get('torch')} | {doc['timestamp']}")
    for run in doc["runs"]:
        if "batch_size" in run:
            print(format_run(run))
        else:
            print(f"{run['precision']:>4} {run['status']}: {run.get('error', '')}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        run_benchmark(
            args.model,
            args.precision,
            args.batch,
            args.prompt_len,
            args.new_tokens,
            args.repeats,
            args.warmup,
            args.seed,
            args.out,
        )
    else:
        cmd_show(args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
