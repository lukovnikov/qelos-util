from ax.service.ax_client import AxClient
import json


class HyperParameter(object):
    def __init__(self, name, **kw):
        super(HyperParameter, self).__init__(**kw)
        self.name = name

    def to_json(self):
        return {
            "name": self.name
        }


class RangeHP(HyperParameter):
    def __init__(self, name, a, b, type="float", log_scale=False, **kw):
        super(RangeHP, self).__init__(name, **kw)
        assert(type in ["float", "int"])
        self.type = type
        self.min = min(a, b)
        self.max = max(a, b)
        self.value_type = type
        self.log_scale = log_scale

    def to_json(self):
        ret = super(RangeHP, self).to_json()
        ret["type"] = "range"
        ret["bounds"] = [self.min, self.max]
        ret["value_type"] = self.type
        ret["log_scale"] = self.log_scale
        return ret


class ChoiceHP(HyperParameter):
    def __init__(self, name, vals, type="float", is_ordered=False, **kw):
        super(ChoiceHP, self).__init__(name, **kw)
        assert(type in ["float", "int", "bool", "str"])
        self.values = vals
        self.type = type
        self.is_ordered = is_ordered

    def to_json(self):
        ret = super(ChoiceHP, self).to_json()
        ret["type"] = "choice"
        ret["values"] = self.values
        ret["value_type"] = self.type
        ret["is_ordered"] = self.is_ordered
        return ret


class FixedHP(HyperParameter):
    def __init__(self, name, value, type="float", **kw):
        super(FixedHP, self).__init__(name, **kw)
        assert(type in ["float", "int", "bool", "str"])
        self.value = value
        self.type = type

    def to_json(self):
        ret = super(FixedHP, self).to_json()
        ret["type"] = "fixed"
        ret["value"] = self.value
        ret["value_type"] = self.type
        return ret


def optimize(parameters=None, evaluation_function=None,
             minimize=True, maxtrials=None, savep=None):
    parameters = [param.to_json() for param in parameters]
    axc = AxClient()
    axc.create_experiment(
        name="_",
        parameters=parameters,
        minimize=minimize,
    )
    print(f"maximum trials: {maxtrials}")
    stop = False
    i = 0
    best_params, values = None, None
    while not stop:
        # print(f"Running trial {i+1}...")
        p, trial_index = axc.get_next_trial()
        print(f"trial index: {trial_index}")
        # print(p)
        axc.complete_trial(trial_index=trial_index, raw_data=evaluation_function(**p))

        best_params, values = axc.get_best_parameters()
        print(f"best params: {best_params}")
        # print(f"values: {values}")
        if savep is not None:
            json.dump(best_params, open(savep, "w"))
        i += 1
        stop = (i >= maxtrials if maxtrials is not None else False)
    best_params, values = axc.get_best_parameters()
    print(f"best params: {best_params}")
    # print(f"values: {values}")
    if savep is not None:
        json.dump(best_params, open(savep, "w"))


