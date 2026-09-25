#!/usr/bin/env python3.11

import unittest

from bounded_launch_revalidation import bounded_revalidate


class FakeClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class BoundedRevalidationTests(unittest.TestCase):
    def test_rebuilds_from_current_time_after_overrun(self):
        observed = []
        plan, started, completed, attempts = bounded_revalidate(
            lambda now: observed.append(now) or {"now": now},
            release_ns=1_000_000_000,
            lower_bound_ns=45_000_000,
            budget_ms=5,
            clock=FakeClock((1_045_000_000, 1_068_000_000,
                             1_068_100_000, 1_069_100_000)),
        )
        self.assertEqual(observed, [45_000_000, 68_100_000])
        self.assertEqual(plan["now"], 68_100_000)
        self.assertEqual((started, completed), (1_068_100_000, 1_069_100_000))
        self.assertEqual([row["within_budget"] for row in attempts], [False, True])

    def test_raises_when_every_attempt_exceeds_budget(self):
        with self.assertRaises(RuntimeError):
            bounded_revalidate(
                lambda now: {"now": now},
                release_ns=0,
                lower_bound_ns=45_000_000,
                budget_ms=5,
                max_attempts=2,
                clock=FakeClock((45_000_000, 51_000_001,
                                 52_000_000, 58_000_001)),
            )


if __name__ == "__main__":
    unittest.main()
