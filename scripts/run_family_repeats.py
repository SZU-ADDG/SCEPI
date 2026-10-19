import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["single", "pair"], required=True)
    parser.add_argument("--thresholds", nargs="+", type=int, choices=[25, 30], default=[30, 25])
    parser.add_argument("--representation", choices=["full", "wt", "identity", "no_cross"], default="full")
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--flexible", action="store_true")
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--no-interaction", action="store_true")
    parser.add_argument("--geometry-intervention", choices=["none", "permuted", "rewired"], default="none")
    parser.add_argument("--objective", choices=["combined", "residual", "phenotype"], default="combined")
    args = parser.parse_args()
    control_tag = (f"_geo{args.geometry_intervention}" if args.geometry_intervention != "none" else "") + ("_n1" if args.no_interaction else "")
    single_representation = "full" if args.representation == "no_cross" else args.representation
    single_grid = json.loads((ROOT / "runs/e3" / f"single_grid_{single_representation}" / "manifest.json").read_text())
    assert single_grid["complete"]
    tag = f"{args.representation}_g{int(args.geometry)}_f{int(args.flexible)}_j{int(args.joint)}{control_tag}"
    if args.objective != "combined":
        tag += "_" + args.objective
    if args.stage == "pair":
        trials = json.loads((ROOT / "runs/e3" / f"pair_grid_{tag}" / "jobs.json").read_text())
        selected = {}
        for threshold in args.thresholds:
            for fold in range(5):
                candidates = [trial for trial in trials if trial["threshold"] == threshold and trial["fold"] == fold]
                assert len(candidates) == 12
                evaluated = []
                for candidate in candidates:
                    manifest = json.loads((ROOT / candidate["run"] / "manifest.json").read_text())
                    assert manifest["complete"] and not manifest["smoke"]
                    evaluated.append({"run": candidate["run"], "validation_score": manifest["validation_score"]})
                selected[f"family{threshold}/fold{fold}"] = min(evaluated, key=lambda item: (item["validation_score"], item["run"]))
        grid = {"selected": selected}
    jobs = []
    for key, selected_single in single_grid["selected"].items():
        if not any(key.startswith(f"family{threshold}/") for threshold in args.thresholds):
            continue
        source_single = json.loads((ROOT / selected_single["run"] / "manifest.json").read_text())
        single_args = source_single["config"]
        if args.stage == "pair":
            selected_pair = grid["selected"][key]
            source_pair = json.loads((ROOT / selected_pair["run"] / "manifest.json").read_text())
            pair_args = source_pair["config"]
        for seed in [11, 23, 47, 89, 131]:
            single_name = f"{single_representation}_p{single_args['projection']}_lr{single_args['learning_rate']:g}_seed{seed}"
            single_run = Path("runs/e3") / key / "single" / single_name
            single_command = [
                sys.executable, "scripts/train_single.py", "--threshold", str(single_args["threshold"]),
                "--fold", str(single_args["fold"]), "--projection", str(single_args["projection"]),
                "--learning-rate", str(single_args["learning_rate"]), "--representation", single_representation,
                "--seed", str(seed),
            ]
            stages = [{"name": "single_training", "command": single_command, "manifest": str(single_run / "manifest.json")}]
            name = key + "/" + single_name
            if args.stage == "pair":
                pair_name = f"{args.representation}_p{pair_args['projection']}_lr{pair_args['learning_rate']:g}_ly{pair_args['lambda_y']:g}_g{int(args.geometry)}_f{int(args.flexible)}_j{int(args.joint)}{control_tag}_{pair_args['objective']}_seed{seed}"
                pair_run = Path("runs/e3") / key / "pair" / pair_name
                pair_command = [
                    sys.executable, "scripts/train_pair.py", "--threshold", str(pair_args["threshold"]), "--fold", str(pair_args["fold"]),
                    "--projection", str(pair_args["projection"]), "--learning-rate", str(pair_args["learning_rate"]),
                    "--lambda-y", str(pair_args["lambda_y"]), "--representation", args.representation,
                    "--objective", pair_args["objective"], "--seed", str(seed), "--single-run", str(single_run),
                    "--geometry-intervention", args.geometry_intervention,
                ]
                for option in ["geometry", "flexible", "joint", "no_interaction"]:
                    if getattr(args, option):
                        pair_command.append("--" + option.replace("_", "-"))
                stages += [{"name": "pair_training", "command": pair_command, "manifest": str(pair_run / "manifest.json")}]
                for partition in ["validation", "test"]:
                    stages.append({
                        "name": "evaluation_" + partition,
                        "command": [sys.executable, "scripts/evaluate_family.py", "--run", str(pair_run), "--partition", partition],
                        "manifest": str(pair_run / ("evaluation_" + partition) / "manifest.json"),
                    })
                name = key + "/" + pair_name
            jobs.append({"name": name, "stages": stages})
    name = f"single_repeats_{single_representation}" if args.stage == "single" else f"pair_repeats_{tag}"
    if set(args.thresholds) != {25, 30}:
        name += "_family" + "_".join(str(value) for value in args.thresholds)
    output = ROOT / "runs/e3" / name
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "pair":
        (output / "selected_hyperparameters.json").write_text(json.dumps(grid["selected"], indent=2))
    run_jobs(ROOT, jobs, output)


if __name__ == "__main__":
    main()
