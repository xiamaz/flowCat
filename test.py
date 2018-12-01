#!/usr/bin/env python3
import argparse
import unittest


def get_args():
    parser = argparse.ArgumentParser(description="Run unittests and integration tests.")
    parser.add_argument("type", choices=["all", "unit", "integration"], help="Select which type of tests to run")
    args = parser.parse_args()
    return args


def suite(pattern="test*.py"):
    """Create a testsuite."""
    return unittest.TestLoader().discover("tests", pattern=pattern)


def run_unit():
    unit_tests = suite("test_*.py")
    unittest.TextTestRunner(verbosity=2).run(unit_tests)


def run_integration():
    integration_tests = suite("testint_*.py")
    unittest.TextTestRunner(verbosity=2).run(integration_tests)


def run_all():
    run_unit()
    run_integration()


def run_choice(runtype):
    if runtype == "all":
        return run_all
    if runtype == "unit":
        return run_unit
    if runtype == "integration":
        return run_integration
    return lambda: print("Invalid choice")


def main():
    args = get_args()
    run_choice(args.type)()


if __name__ == "__main__":
    main()
