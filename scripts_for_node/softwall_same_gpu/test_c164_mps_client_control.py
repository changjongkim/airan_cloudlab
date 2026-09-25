#!/usr/bin/env python3.11

import unittest

from c164_mps_client_control import parse_clients


class MpsClientControlTests(unittest.TestCase):
    def test_parse_legacy_ps(self):
        raw = """PID ID SERVER DEVICE NAMESPACE COMMAND
9741 0 6472 GPU-cb12 4026531836 python3 worker.py --worker-epoch e1
9743 1 6472 GPU-cb12 4026531836 python3 mandatory.py
"""
        rows = parse_clients(raw)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["pid"], 9741)
        self.assertEqual(rows[0]["server_pid"], 6472)
        self.assertIn("worker-epoch", rows[0]["command"])

    def test_ignore_headers_and_errors(self):
        self.assertEqual(parse_clients("no clients\n"), [])


if __name__ == "__main__":
    unittest.main()
