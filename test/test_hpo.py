import random
from unittest import TestCase
import qelos as q


class TestRunExperiments(TestCase):
    def test_run_experiments(self):
        def run(**kw):
            ret = random.random()
            print(ret)
            return {"ret": ret}
        q.hpo.run_experiments(run, ranges={"lr": [1, 2, 3, 4, 5]},
                              decision_field="ret", stop_value="<.5")
