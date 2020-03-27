import json
import os
import random
import shutil
from collections import OrderedDict
from itertools import product

import torch
import qelos as q


__all__ = ["run_hpo_cv", "run_experiments"]

import ujson


def run(**kw):
    print(kw)
    return random.random()


def run_experiments(runf, ranges, path=None, decision_field="valid_acc", stop_value:str=None,
                    **kw):
    tt = q.ticktock("HPO")
    tt.msg("running experiments")
    tt.msg(ujson.dumps(ranges, indent=4))
    _ranges = [[(k, v) for v in ranges[k]] for k in ranges]
    all_combos = list(product(*_ranges))
    random.shuffle(all_combos)
    tt.msg(f"Number of possible combinations: {len(all_combos)}")
    specs = [dict(x) for x in all_combos]

    results = []

    for spec in specs:
        do_stop = False
        tt.msg(f"Training for specs: {spec}")
        kw_ = kw.copy()
        kw_.update(spec)
        result = runf(**kw_)

        decision_field_value = None
        if decision_field in result:
            decision_field_value = float(result[decision_field])
        if stop_value is not None:
            assert(len(stop_value > 2) and stop_value[0] in "<>")
            less_or_greater = stop_value[0]
            less_or_greater = 1 if less_or_greater == ">" else -1
            value = float(stop_value[1:])
            if decision_field_value * less_or_greater > value * less_or_greater:
                do_stop = True
                tt.msg(f"\"{decision_field}\" reached a value of {decision_field_value:.3f}, which is {stop_value}.\n Stopping further experiments.")

        results.append(result)
        if path is not None:
            with open(path, "w") as f:
                ujson.dump(results, f, indent=4)

        if do_stop:
            break
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