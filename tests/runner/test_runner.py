import json
import os
import os.path as path
import re
import subprocess

import pytest

from shepherd.comm import *
from shepherd.constants import OUTPUT_DIR, DEFAULT_OUTPUT_FILE
from shepherd.runner import *


def start_cli(command, mocker):
    handle = subprocess.Popen(command)
    return handle.kill


# TODO add some asyncio runner, using a background thread with a separate event loop might also be feasible
@pytest.mark.skip(reason="freezes circleci")
@pytest.mark.parametrize('start', (start_cli,))
async def test_runner(job, feeding_socket, runner_setup, mocker, start):  # for coverage reporting
    socket, port = feeding_socket
    job_id, job_dir = job
    version, stream, expected = runner_setup
    base_config_path = path.join('examples', 'docker', 'emloop_example', 'emloop-test', version)

    # test both config by dir and config by file
    for config_path in [base_config_path, path.join(base_config_path, 'config.yaml')]:
        command = ['shepherd-runner', '-p', str(port), '-s', stream, config_path]
        killswitch = start(command, mocker)
        await Messenger.send(socket, InputMessage(dict(job_id=job_id, io_data_root=job_dir)))
        await Messenger.recv(socket, [DoneMessage])
        killswitch()  # terminate the runner
        output = json.load(open(path.join(job_dir, job_id, OUTPUT_DIR, DEFAULT_OUTPUT_FILE)))
        assert output['output'] == [expected]


def test_n_gpus(mocker):
    n_system_gpus = len([s for s in os.listdir("/dev") if re.search(r'nvidia[0-9]+', s) is not None])
    assert n_available_gpus() == n_system_gpus
    mocker.patch('os.environ', {'NVIDIA_VISIBLE_DEVICES': '0,3'})
    assert n_available_gpus() == 2
    mocker.patch('os.environ', {'CUDA_VISIBLE_DEVICES': '1'})
    assert n_available_gpus() == 1
    mocker.patch('os.environ', {'NVIDIA_VISIBLE_DEVICES': '0,3', 'CUDA_VISIBLE_DEVICES': ''})
    assert n_available_gpus() == 0
