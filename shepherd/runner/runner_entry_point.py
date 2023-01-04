import asyncio
import importlib
import logging
import os
import sys
from argparse import ArgumentParser

__all__ = ['main']
from typing import Tuple, Optional

from shepherd.runner import BaseRunner


def create_argparser():
    """Create and return argument parser."""
    parser = ArgumentParser('shepherd runner')
    parser.add_argument('-p', '--port', dest="port", default=9999, type=int, help='Socket port to bind to')
    parser.add_argument('-s', '--stream', default='predict', help='Dataset stream name')
    parser.add_argument('-r', '--runner', default='shepherd.runner.JSONRunner', help='Fully qualified runner class')
    parser.add_argument('config_path', help='configuration file path')
    return parser


def main() -> None:
    """
    Create a runner and list on the configured port for job ``InputMessage`` s.

    Can be invoked with installed ``shepherd-runner`` command.
    """

    # parse args
    sys.path.insert(0, os.getcwd())
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s-%(levelname)s-%(name)s::%(module)s|%(lineno)s:: %(message)s")

    args = create_argparser().parse_args()

    module_name = "model.isletnet_runner"
    class_name = "IsletnetRunner"

    _module = importlib.import_module(module_name)
    runner = getattr(_module, class_name)(args.config_path, args.port, args.stream)

    # listen for input messages
    asyncio.run(runner.process_all())


if __name__ == '__main__':
    main()  # pragma: no cover
