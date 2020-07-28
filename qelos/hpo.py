import json
import math
import os
import random
import shutil
import string
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from typing import Callable

import optuna
import torch
from optuna import Trial

import qelos as q
import numpy as np

__all__ = ["run_hpo_cv", "run_experiments", "run_experiments_optuna", "run_experiments_custom_genetic"]

import ujson

import matplotlib as mpl
from matplotlib import pyplot as plt


def run(**kw):
    print(kw)
    return random.random()


def test_optuna():
    import optuna
    uniq = set()
    def obj(trial:Trial):
        x = trial.suggest_discrete_uniform("x", -5, 6, 1) #suggest_uniform("x", -5, 5)
        y = trial.suggest_discrete_uniform("y", -5, 6, 1)
        setting = f"{x}-{y}"
        if setting in uniq:
            raise optuna.TrialPruned()
        if not(abs(x) > abs(y) or y < 0):
            raise optuna.TrialPruned()
            # raise Exception("invalid config")
        uniq.add(setting)
        return (abs(x) + abs(y)) ** 2

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=500, catch=(Exception,))

    print(study.best_params)
    print(study.best_value)

    # ts = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ts = [t for t in study.trials]
    x = [t.params["x"] for t in ts]
    y = [t.params["y"] for t in ts]
    # plt.scatter([t.params["x"] for t in ts], [t.params["y"] for t in ts])
    # plt.show()

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()


def test_hyperopt():
    from hyperopt import hp, fmin
    import hyperopt

    ranges = {
        "x": [-3, -2, -1, 0, 1, 2, 3],
        "y": [-3, -2, -1, 0, 1, 2, 3]
    }

    space = {k: hp.choice(k, v) for k, v in ranges.items()}

    uniq = set()
    def obj(d):
        x, y = d["x"], d["y"]
        setting = f"{x}-{y}"
        status = hyperopt.STATUS_OK
        if setting in uniq:
            status = hyperopt.STATUS_FAIL
        if not(abs(x) > abs(y) or y < 0):
            status = hyperopt.STATUS_FAIL
        if status == hyperopt.STATUS_OK:
            uniq.add(setting)
        return {"loss": (abs(x) + abs(y)) ** 2, "status": status}

    trials = hyperopt.Trials()
    best = fmin(obj, space, hyperopt.tpe.suggest, max_evals=200, trials=trials)
    statuses = [t["result"]["status"] for t in trials.trials]
    ts = zip(trials.vals["x"], trials.vals["y"], statuses)
    # ts = [t for t in ts if t[2] == "ok"]
    xs, ys, stats = zip(*ts)
    xs = [ranges["x"][x] for x in xs]
    ys = [ranges["y"][y] for y in ys]

    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

    print(best)
    # plt.scatter([t["misc"]['vals']["x"][0] for t in trials], [t["misc"]['vals']["y"][0] for t in trials])
    # plt.show()


def test_custom_genetic():
    ranges = {"x": list(range(-3, 4)),
              "y": list(range(-3, 4)),
              "z": list(range(-3, 4)),
              "a": list(range(-2, 3)),
              "b": list(range(-2, 3)),
              "c": list(range(-2, 3))}
    def checkconfig(spec):
        return spec["x"] + spec["y"] <= 0

    def run(x=0, y=0, z=0, a=0, b=0, c=0):
        ret = (abs(x) + abs(y) + abs(z) + abs(a) + abs(b) + abs(c)) ** 2
        return {"loss": ret}

    run_experiments_custom_genetic(run, "loss", ranges, None, checkconfig)



def run_experiments_custom_genetic(runf, optout, ranges, path_prefix, check_config:Callable=None,
                           minimize=True, minimum_init_coverage=1, population_size=10, no_duplicates=True, eps=0.1, **kw):
    tt = q.ticktock("HPO")
    tt.msg("(using custom evolutionary HPO)")
    tt.msg("running experiments")
    tt.msg(ujson.dumps(ranges, indent=4))
    _ranges = [[(k, v) for v in ranges[k]] for k in ranges]
    all_combos = list(product(*_ranges))
    random.shuffle(all_combos)
    specs = [dict(x) for x in all_combos]
    specschema = sorted(specs[0].keys())
    specs = {tuple([x[k] for k in specschema]) for x in specs}
    if check_config is not None:
        specs = {spec for spec in specs if check_config(dict(zip(specschema, spec)))}
    tt.msg(f"Number of possible combinations: {len(specs)}")
    specs = list(specs)

    rand = "".join(random.choice(string.ascii_letters) for i in range(6))
    path = path_prefix + f".{rand}.xps" if path_prefix is not None else None
    f = None
    if path is not None:
        f = open(path, "w")

    revind = {k: {} for k in ranges}
    for i, spec in enumerate(specs):
        for k, v in zip(specschema, spec):
            if f"{v}" not in revind[k]:
                revind[k][f"{v}"] = set()
            revind[k][f"{v}"].add(i)

    initpop = []
    # initcounts = {f"{k}:{v}": 0 for v in revind[k] for k in revind}
    initcounts = {k: {ve: 0 for ve in revind[k]} for k in revind}

    cnt = 0

    while min([min([initcounts[k][v] for v in initcounts[k]]) for k in initcounts]) < minimum_init_coverage:
        bestspec = None
        bestspecscore = 0
        bestpossiblespecscore = len(revind)
        for i, spec in enumerate(specs):
            if spec in initpop:
                continue
            specscore = 0
            for k, v in zip(specschema, spec):
                specscore += max(0, minimum_init_coverage - initcounts[k][f"{v}"])
            if specscore > bestspecscore:
                bestspec = spec
                bestspecscore = specscore
            if bestspecscore >= bestpossiblespecscore:
                break
        if bestspec is None:
            raise Exception("something wrong")
        initpop.append(bestspec)
        for k, v in zip(specschema, bestspec):
            initcounts[k][f"{v}"] += 1

    # print(initpop)
    tt.msg(f"Initial population of {len(initpop)} made. Evaluating...")

    remainingspecs = set(deepcopy(specs))

    results = []
    for spec in initpop:
        try:
            tt.msg(f"Training for specs: {spec} (#{len(results)})")
            kw_ = kw.copy()
            kw_.update(dict(zip(specschema, spec)))
            try:
                result = runf(**kw_)
            except Exception as e:
                print("EXCEPTION!")
                print(e)
                # raise e
                result = kw_
                result.update({"type": "EXCEPTION", "exception": str(e)})

            results.append(result)
            remainingspecs -= {spec,}
            cnt += 1
            if f is not None:
                ujson.dump(results, f, indent=4)
        except RuntimeError as e:
            print("Runtime error. Probably CUDA out of memory.\n...")

    pop = list(zip(initpop, [res[optout] for res in results]))
    print(pop)

    while len(remainingspecs) > 0:
        rankedpop = sorted(pop, key=lambda x: x[-1], reverse=not minimize)
        pop = rankedpop[:population_size]
        validandunique = False
        spec = None
        while not validandunique:
            if random.random() < eps:
                spec = random.choice(list(remainingspecs))
                validandunique = True
            else:
                a = random.choice(pop)
                b = random.choice(pop)
                recomb = [False for _ in range(len(a))]
                while all([recomb[i] == recomb[0] for i in range(len(recomb))]):
                    recomb = [random.choice([True, False]) for _ in range(len(a[0]))]
                # random.shuffle(recomb)
                spec = tuple([a[0][i] if recomb[i] else b[0][i] for i in range(len(a[0]))])
                validandunique = spec in remainingspecs
        try:
            tt.msg(f"Training for specs: {spec} (#{len(results)})")
            kw_ = kw.copy()
            kw_.update(dict(zip(specschema, spec)))
            try:
                result = runf(**kw_)
            except Exception as e:
                print("EXCEPTION!")
                print(e)
                # raise e
                result = kw_
                result.update({"type": "EXCEPTION", "exception": str(e)})

            results.append(result)
            remainingspecs -= {spec,}
            pop.append((spec, result[optout]))
            cnt += 1
            if f is not None:
                ujson.dump(results, f, indent=4)
        except RuntimeError as e:
            print("Runtime error. Probably CUDA out of memory.\n...")
        print(pop)
        print()
    # print(json.dumps(results, indent=4))
    # print(len(results))



def run_experiments_optuna(runf, optout, ranges, path_prefix, check_config:Callable=None,
                           minimize=True,
                           **kw):
    tt = q.ticktock("HPO")
    tt.msg("running experiments")
    tt.msg(ujson.dumps(ranges, indent=4))
    _ranges = [[(k, v) for v in ranges[k]] for k in ranges]
    all_combos = list(product(*_ranges))
    random.shuffle(all_combos)
    specs = [dict(x) for x in all_combos]
    tt.msg(f"Number of possible combinations: {len(specs)}")

    rand = "".join(random.choice(string.ascii_letters) for i in range(6))
    path = path_prefix + f".{rand}.xps"

    f = None
    if path is not None:
        f = open(path, "w")

    results = []

    # convert ranges to optuna
    def obj(trial:optuna.Trial):
        spec = {}
        for k, v in ranges.items():
            spec[k] = trial.suggest_categorical(k, v)

        if not check_config(spec):
            raise optuna.TrialPruned()

        tt.msg(f"Training for specs: {spec}")
        kw_ = kw.copy()
        kw_.update(spec)
        hadexception = False
        try:
            result = runf(**kw_)
        except Exception as e:
            print("EXCEPTION!")
            print(e)
            result = kw_
            result.update({"type": "EXCEPTION", "exception": str(e)})
            hadexception = True

        results.append(result)
        if f is not None:
            ujson.dump(results, f, indent=4)
            f.flush()

        if hadexception:
            raise optuna.TrialPruned()
        return result[optout]

    f.close()

    study = optuna.create_study(direction="minimize" if minimize else "maximize")
    study.optimize(obj, n_trials=len(specs), catch=(Exception,))

    return study


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
    if len(specs) < 1:
        raise Exception("No combination is possible!")

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
                print("EXCEPTION!")
                print(e)
                # raise e
                result = kw_
                result.update({"type": "EXCEPTION", "exception": str(e)})

            results.append(result)
            if path is not None:
                with open(path, "w") as f:
                    ujson.dump(results, f, indent=4)

            if pmtf is not None and pmtf(result):
                tt.msg(f"Criterion satisfied.\nStopping further experiments.")
                break
        except RuntimeError as e:
            print("Runtime error. Probably CUDA out of memory.\n...")
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
    test_custom_genetic()
    # q.argprun(do_run_hpo_cv)
    # test_hyperopt()
    # test_optuna()