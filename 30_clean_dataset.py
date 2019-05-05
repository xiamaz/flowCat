#!/usr/bin/env python3
"""
Move the cleaning out of the script running every time and generate cleaned
data instead.

Clean the provided input json dataset in order to:
1. Remove case ids that are in two different cohorts.
2. Remove duplicated or faulty FCS samples from cases.

Optional goals:
- Provide SQLITE database with samples
"""
import flowcat
