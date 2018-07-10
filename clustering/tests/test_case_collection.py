"""Test creation of collections and collection views."""
from .base import BaseCollection, BaseView, resolve_path


class CaseCollectionTest(BaseCollection):
    """Test parsing of case_info files and creation of case views."""

    def test_unique_tubes(self):
        """Ensure that the correct number of unique tubes is detected from the
        imported case json data."""
        self.assertListEqual(
            self.collection.tubes, [1, 2, 3],
            "List of tubes are not equal."
        )

    def test_create_view(self):
        """Create case views with different case parameters."""
        all_view = self.collection.create_view()
        # assert that all cohorts are included
        self.assertEqual(len(all_view._data), 10, "Mismatch in cohort number")

        self.assertEqual(
            sum(map(len, all_view._data.values())), 64, "Mismatch case number"
        )

        simple_view = self.collection.create_view(
            num=1
        )
        self.assertEqual(
            sum(map(len, simple_view._data.values())), len(simple_view._data),
            "Single selection does not match number of cohorts."
        )

        groups_view = self.collection.create_view(
            groups=["CLL"]
        )
        self.assertEqual(
            list(groups_view._data.keys()), ["CLL"],
            "Group selection does not match seletected groups."
        )

        tubes_view = self.collection.create_view(
            tubes=[1]
        )
        self.assertTrue(
            all([
                any(
                    int(path["tube"]) == 1
                    for path in case["destpaths"]
                )
                for cases in tubes_view._data.values()
                for case in cases
            ]),
            "Cases with only selected tube are included in cohort."
        )

        test_label = "289a87194c2e03ce46d5fd55b2ab7ca10b6d757e"
        labels_view = self.collection.create_view(
            labels=[test_label]
        )

        self.assertEqual(
            sum(map(len, labels_view._data.values())), 1
        )
        self.assertTrue(
            all([
                case["id"] == test_label
                for cases in labels_view._data.values()
                for case in cases
            ]),
            "Selected label not found in view."
        )


class CaseViewTest(BaseView):
    """Test file loading and operations on CaseViews."""
    def test_yield_data(self):
        tube_tests = [(1, 62), (2, 63), (3, 62)]
        for tube, num in tube_tests:
            with self.subTest(tube):
                self.assertEqual(
                    len([d for d, _ in self.view.yield_data(tube=tube)]),
                    num
                )
