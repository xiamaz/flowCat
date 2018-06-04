"""Test different clustering functions on correctness."""
import pandas as pd
from clustering.clustering import (
    create_pipeline, create_pipeline_double_som, create_pipeline_multistage
)
from .test_case_collection import BaseView


class ClusteringTest(BaseView):

    def _yield_batch(self, tube):
        return pd.concat([d for _, d in self.view.yield_data(tube)])

    def test_pipeline(self):
        pipelines = [
            ("normal", create_pipeline),
            ("double_som", create_pipeline_double_som),
            ("multistage", create_pipeline_multistage)
        ]
        for name, pipefunc in pipelines:
            with self.subTest(name):
                pipe = pipefunc()
                pipe.fit(self._yield_batch(1))
                results = [
                    pipe.transform(d) for _, d in self.view.yield_data(1)
                ]
                print(results)
