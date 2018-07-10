"""Test different clustering functions on correctness."""
import pandas as pd
from clustering.clustering import (
    create_pipeline, create_pipeline_multistage
)
from .base import BaseView


class ClusteringTest(BaseView):

    def setUp(self):
        super().setUp()
        self.small_view = self.collection.create_view(
            num=1, groups=["CLL", "normal"],
            tmpdir=self.tmppath
        )

    def _yield_batch(self, tube):
        return pd.concat([d for _, d in self.small_view.yield_data(tube)])

    def test_pipeline(self):
        pipelines = [
            ("normal", create_pipeline),
            ("multistage", create_pipeline_multistage)
        ]
        for name, pipefunc in pipelines:
            with self.subTest(name):
                pipe = pipefunc()
                pipe.fit(self._yield_batch(1))
                results = [
                    pipe.transform(d) for _, d in self.small_view.yield_data(1)
                ]
                for result in results:
                    print(result)
                    self.assertEqual(result.shape, (100,))
