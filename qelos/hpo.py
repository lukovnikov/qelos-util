import json
import os
import random
import shutil
import string
from collections import OrderedDict
from itertools import product
from typing import Callable

import torch
import qelos as q


__all__ = ["run_hpo_cv", "run_experiments"]

import ujson


def run(**kw):
    print(kw)
    return random.random()


def run_experiments(runf, ranges, path=None, path_prefix=None, check_config:Callable=None,
                    pmtf:Callable=None,
                    **kw):
    """

    :param runf:
    :param ranges:      dict of hyperparam ranges
    :param path:
    :param pmtf:        PreMature Stopping Function. Receives outputs from runf. Must return bool.
                        If returns True, current set of experiments is terminated.
    :param kw:
    :return:
    """
    tt = q.ticktock("HPO")
    tt.msg("running experiments")
    tt.msg(ujson.dumps(ranges, indent=4))
    _ranges = [[(k, v) for v in ranges[k]] for k in ranges]
    all_combos = list(product(*_ranges))
    random.shuffle(all_combos)
    specs = [dict(x) for x in all_combos]
    if check_config is not None:
        specs = [spec for spec in specs if check_config(spec)]
    tt.msg(f"Number of possible combinations: {len(specs)}")

    results = []

    if path is None:
        rand = "".join(random.choice(string.ascii_letters) for i in range(6))
        path = path_prefix + f".{rand}.xps"
    else:
        print("Warning: deprecated, use path_prefix instead.")

    for spec in specs:
        try:
            tt.msg(f"Training for specs: {spec}")
            kw_ = kw.copy()
            kw_.update(spec)
            try:
                result = runf(**kw_)
            except Exception as e:
                result = {"type": "EXCEPTION", "exception": str(e)}

            results.append(result)
            if path is not None:
                with open(path, "w") as f:
                    ujson.dump(results, f, indent=4)

            if pmtf is not None and pmtf(result):
                tt.msg(f"Criterion satisfied.\nStopping further experiments.")
                break
        except RuntimeError as e:
            print("Runtime error. Probably CUDA out of memory")
    return results


def run_hpo_cv(runf, ranges, numcvfolds=6, path=None, **kw):
    """

    :param runf:            must have kwargs "testfold", "numcvfolds"
    :param ranges:
    :param numcvfolds:
    :param path:
    :param kw:
    :return:
    """
    tt = q.ticktock("HPO")
    tt.msg(ujson.dumps(ranges, indent=4))
    _ranges = [[(k, v) for v in ranges[k]] for k in ranges]
    all_combos = list(product(*_ranges))
    random.shuffle(all_combos)
    tt.msg(f"Number of possible combinations: {len(all_combos)}")
    specs = [dict(x) for x in all_combos]

    results = OrderedDict({"best": (None, -1e9), "all": []})

    for spec in specs:
        scores = []
        tt.msg(f"Training for specs: {spec}")
        for i in range(numcvfolds):
            kw_ = kw.copy()
            kw_.update(spec)
            score = runf(testfold=i, numcvfolds=numcvfolds, **kw_)
            tt.msg(f"SCORE: {score} FOR FOLD {i+1}/{numcvfolds}")
            scores.append(score)
        score = sum(scores)/len(scores)
        tt.msg(f"AVERAGE SCORE OVER FOLDS: {score}")
        tt.msg(f"For config: {spec}")
        results["all"].append((spec, score))
        if results["best"][1] < score:
            results["best"] = (spec, score)
        if path is not None:
            with open(path, "w") as f:
                ujson.dump(results, f, indent=4)
    return results


def do_run_hpo_cv(numcvsplits=6, cuda=False, gpu=0):
    ranges = {"encdim": [128, 256, 400, 512],
              "dropout": [.2, .35, .5],
              "smoothing": [0., .1, .2],
              "epochs": [1, 2]
              }
    run_hpo_cv(run, ranges, numcvsplits=numcvsplits, path=__file__+".hpo", cuda=cuda, gpu=gpu)


if __name__ == '__main__':
    q.argprun(do_run_hpo_cv)