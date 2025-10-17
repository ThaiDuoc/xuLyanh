import argparse
import inspect
import sys
import unittest
from types import ModuleType
from typing import Any, List

#!/usr/bin/env python3
"""
tesst.py - lightweight test runner / inspector

Usage examples:
    python tesst.py --self-test
    python tesst.py --path ../mymodule.py --list
    python tesst.py --path ../mymodule.py --call my_function arg1 arg2
"""

import importlib.util


def load_module_from_path(path: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location("target_module", path)
        if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return module


def list_callables(module: ModuleType) -> List[str]:
        return [
                name
                for name, obj in inspect.getmembers(module, inspect.isfunction)
                if obj.__module__ == module.__name__
        ]


def call_function(module: ModuleType, func_name: str, args: List[str]) -> Any:
        func = getattr(module, func_name, None)
        if func is None or not callable(func):
                raise AttributeError(f"No function '{func_name}' in module")
        # try to convert numeric args to int/float
        parsed_args = []
        for a in args:
                for conv in (int, float):
                        try:
                                parsed_args.append(conv(a))
                                break
                        except Exception:
                                continue
                else:
                        parsed_args.append(a)
        return func(*parsed_args)


class SelfTests(unittest.TestCase):
        def test_arithmetic(self):
                self.assertEqual(1 + 1, 2)

        def test_string(self):
                self.assertTrue("hi".upper() == "HI")


def main(argv=None):
        parser = argparse.ArgumentParser(description="Simple test/inspect utility")
        parser.add_argument("--self-test", action="store_true", help="run built-in unit tests")
        parser.add_argument("--path", help="path to python file to inspect")
        parser.add_argument("--list", action="store_true", help="list top-level functions in path")
        parser.add_argument("--call", nargs="+", help="call function: --call funcname [args...]")
        args = parser.parse_args(argv)

        if args.self_test:
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests)
                runner = unittest.TextTestRunner(verbosity=2)
                return 0 if runner.run(suite).wasSuccessful() else 1

        if args.path:
                try:
                        mod = load_module_from_path(args.path)
                except Exception as e:
                        print(f"Error loading module: {e}", file=sys.stderr)
                        return 2

                if args.list:
                        names = list_callables(mod)
                        if not names:
                                print("No top-level functions found.")
                        else:
                                for n in names:
                                        print(n)
                        return 0

                if args.call:
                        func_name, *func_args = args.call
                        try:
                                result = call_function(mod, func_name, func_args)
                                print("Result:", result)
                        except Exception as e:
                                print(f"Error calling function: {e}", file=sys.stderr)
                                return 3
                        return 0

                print("Module loaded. Use --list to see functions or --call to invoke one.")
                return 0

        parser.print_help()
        return 0


if __name__ == "__main__":
        raise SystemExit(main())