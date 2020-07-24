import json
import os
import random
import shutil
import string
from collections import OrderedDict
from itertools import product
from typing import Callable

import optuna
import torch
import qelos as q


__all__ = ["run_hpo_cv", "run_experiments"]

import ujson
from hyperopt import hp, fmin
import hyperopt

import matplotlib as mpl
from matplotlib import pyplot as plt


def run(**kw):
    print(kw)
    return random.random()


def test_optuna():
    import optuna
    def obj(trial):
        x = trial.suggest_uniform("x", -5, 5)
        y = trial.suggest_uniform("y", -5, 5)
        if not(abs(x) > abs(y) or y < 0):
            # raise optuna.TrialPruned()
            raise Exception("invalid config")
        return (abs(x) + abs(y)) ** 2

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=1000, catch=(Exception,))

    print(study.best_params)
    print(study.best_value)


    plt.scatter([t.params["x"] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], [t.params["y"] for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    plt.show()



def test_hyperopt():
    from hyperopt.pyll.stochastic import sample
    ranges = {
        "dropout": hp.choice("dropout", [0., .1, .25, .5, .75]),
        "dropoutemb": hp.choice("dropout", [0., 0.1]),
        "epochs": [50, 100],
        "batsize": hp.choice("batsize", [
            (20, {"maxrdim": hp.choice("maxrdim", [100, 300])}),
            (100, {"maxrdim": hp.choice("maxrdim", [100])})
        ])
    }
    print(sample(ranges))
    # def obj(d):
    #     x, y = d["x"], d["y"]
    #     ret = (abs(x) + abs(y)) ** 2
    #     return {"loss": ret, "status": hyperopt.STATUS_OK}
    #
    # space = {"x": hp.uniform("x", -10, 10), "y": hp.choice("y", [k - 10 for k in range(0, 20)])}
    #
    # trials = hyperopt.Trials()
    #
    # best = fmin(obj, space, hyperopt.tpe.suggest, max_evals=100, trials=trials)
    # print(best)
    # print(trials)
    # plt.scatter([t["misc"]['vals']["x"][0] for t in trials], [t["misc"]['vals']["y"][0] for t in trials])
    # plt.show()


def run_experiments_optuna(runf, optout, ranges, path_prefix, check_config:Callable=None, minimize=True, **kw):
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

        result = runf(**kw_)

        for k, v in result.items():
            trial.set_user_attr(k, v)

        if path is not None:
            with open(path, "w") as f:
                ujson.dump(results, f, indent=4)

        return result[optout]

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
    # q.argprun(do_run_hpo_cv)
    # test_hyperopt()
    test_optuna()