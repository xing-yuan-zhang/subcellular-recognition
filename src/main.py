#!/usr/bin/env python3
# Auto-converted from Jupyter notebook: main (1).ipynb

import os, sys


def _add_src_to_path():
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(6):
        src = os.path.join(cur, "src")
        if os.path.isdir(src):
            if src not in sys.path:
                sys.path.insert(0, src)
            return
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt

_add_src_to_path()

def main():
    import os
    import sys
    import subprocess

    import json

    import config
    from training import train_single_model, train_all_organelle_models
    from inference import evaluate_on_organelle, cross_eval_matrix

    model_name = "resnet18"
    organelle = "focal_adhesion"

    metrics_single = train_single_model(
        model_name=model_name,
        organelle=organelle,
        pretrained=True,
    )

    eval_self = evaluate_on_organelle(
        model_name=model_name,
        train_organelle=organelle,
        test_organelle=organelle,
    )

    cross_results = cross_eval_matrix(
        model_name=model_name,
        organelles=config.ORGANELLES,
    )

    results_all = train_all_organelle_models(
        model_names=config.MODEL_NAMES,
        organelles=config.ORGANELLES,
        pretrained=True,
    )

    with open("cross_eval_{}_{}.json".format(model_name, "_".join(config.ORGANELLES)), "w") as f:
        json.dump(cross_results, f, indent=2)


if __name__ == '__main__':
    main()

