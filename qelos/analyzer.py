from typing import List, Callable, Union

import fire
# from IPython import embed
import ujson as json
from fnmatch import fnmatch

from traitlets.config import get_config

SEED_FIELD = "seed"
NUMSEEDS_FIELD = "numseeds"
OUTCOME_FIELDS = "*_acc,*_loss"

def analyze(p="",
            seed_field=SEED_FIELD,
            numseeds_field=NUMSEEDS_FIELD,
            outcome_fields=OUTCOME_FIELDS,
            ):
    import numpy as np
    import pandas as pd
    outcome_fields = set(outcome_fields.split(","))

    def is_outcome_field(_fieldname):
        return any([fnmatch(_fieldname, outcome_field_pattern) for outcome_field_pattern in outcome_fields])

    print(f"Loading {p}")
    with open(p) as f:
        data = json.load(f)
        if len(data) > 0:
            columns = {}
            # sync schemas
            cols = []
            for d in data:
                for k in d.keys():
                    if k not in cols:
                        cols.append(k)
            for col in cols:
                columns[col] = []
            # populate columns variable
            for d in data:
                for k in cols:
                    if k not in d:
                        v = None
                    else:
                        v = d[k]
                    columns[k].append(v)
            # l = len(data)
            # # group by hyperparam setting, extract same settings and present
            # if seed_field in columns:   # then average over seeds
            #     newdata = {}
            #     newdata[numseeds_field] = 0
            #     configs = {}
            #     for i in range(l):
            #         config = [(k, v[i]) for k, v in columns.items() if (k not in outcome_fields and k != seed_field)]
            #         if config not in configs:
            #             configs[config] = []
            #         configs[config] = i
            #     for config, indexes in configs.items():
            #         for k, v in config:
            #             newdata[k] = v
            #         for k, v in columns.items():
            #             if k not in newdata:
            #                 newdata[k] = 0
    doesntmatter = {"gpu",}
    df = pd.DataFrame.from_dict(columns)
    # extract same settings from table TODO?
    sames = {}
    for c in df.columns:
        if len(set(df[c])) == 1:
            sames[c] = list(set(df[c]))[0]
            del df[c]
    for c in doesntmatter:
        if c in df.columns:
            del df[c]

    # average over seeds
    if seed_field in df.columns:
        others = [item for item in df.columns if item != seed_field and not is_outcome_field(item)]
        if len(others) > 0:
            df = df.groupby(by=others, as_index=False).mean()
        else:
            df = df.mean()
        # del df[seed_field]

    def group_by(_df:pd.DataFrame, by:str, avg_cols:Union[List[str], Callable]):
        if by not in _df.columns:
            raise Exception(f"'by' argument column {by} not in '_df's columns")
        groupbys = df.columns
        if isinstance(avg_cols, Callable):
            groupbys = [col for col in groupbys if not avg_cols(col)]
        else:
            groupbys = [col for col in groupbys if col not in avg_cols]
        groupbys = [col for col in groupbys if col != by]
        ret = _df.groupby(by=groupbys, as_index=False)
        return ret

    c = get_config()
    c.InteractiveShellEmbed.colors = "Linux"
    embed(config=c)


if __name__ == '__main__':
    fire.Fire(analyze)