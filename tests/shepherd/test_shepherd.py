import json

import gevent
import pytest


from cxworker.sheep import BareSheep, DockerSheep, SheepConfigurationError
from cxworker.shepherd import Shepherd
from cxworker.api.errors import UnknownSheepError, UnknownJobError
from cxworker.shepherd.config import WorkerConfig
from cxworker.utils.storage import minio_object_exists


def test_shepherd_init(valid_config: WorkerConfig, minio):
    shepherd = Shepherd(valid_config.sheep, valid_config.data_root, minio, valid_config.registry)

    assert isinstance(shepherd['bare_sheep'], BareSheep)

    with pytest.raises(UnknownSheepError):
        _ = shepherd['UnknownSheep']
    shepherd.notifier.close()

    valid_config.sheep['docker_sheep'] = {'port': 9002, 'type': 'docker'}
    shepherd = Shepherd(valid_config.sheep, valid_config.data_root, minio, valid_config.registry)
    assert isinstance(shepherd['docker_sheep'], DockerSheep)
    shepherd.notifier.close()

    with pytest.raises(SheepConfigurationError):
        shepherd = Shepherd(valid_config.sheep, valid_config.data_root, minio)  # missing docker registry

    shepherd.notifier.close()

    valid_config.sheep['my_sheep'] = {'type': 'unknown'}
    with pytest.raises(SheepConfigurationError):
        Shepherd(valid_config.sheep, valid_config.data_root, minio, valid_config.registry)  # unknown sheep type


def test_shepherd_status(shepherd):
    sheep_name, sheep = next(shepherd.get_status())
    assert not sheep.running
    assert sheep_name == 'bare_sheep'


def test_job(shepherd, job, minio):
    job_id, job_meta = job

    with pytest.raises(UnknownJobError):
        shepherd.is_job_done(job_id)

    shepherd.enqueue_job(job_id, job_meta)
    assert not shepherd.is_job_done(job_id)
    gevent.sleep(1)
    assert shepherd['bare_sheep'].running
    assert shepherd.is_job_done(job_id)
    assert minio_object_exists(minio, job_id, 'outputs/output.json')
    assert minio_object_exists(minio, job_id, 'done')
    output = json.loads(minio.get_object(job_id, 'outputs/output.json').read().decode())
    assert output['key'] == [1000]
    assert output['output'] == [1000*2]


def test_failed_job(shepherd, bad_job, minio):
    job_id, job_meta = bad_job
    shepherd.enqueue_job(job_id, job_meta)  # runner should fail to process the job (and send an ErrorMessage)
    gevent.sleep(1)
    assert shepherd['bare_sheep'].running
    assert shepherd.is_job_done(job_id)
    assert minio_object_exists(minio, job_id, 'error')


def test_bad_configuration_job(shepherd, bad_configuration_job, minio):
    job_id, job_meta = bad_configuration_job
    shepherd.enqueue_job(job_id, job_meta)  # shepherd should get SheepConfigurationError
    gevent.sleep(1)
    assert not shepherd['bare_sheep'].running
    assert shepherd.is_job_done(job_id)
    assert minio_object_exists(minio, job_id, 'error')


def test_bad_runner_job(shepherd, bad_runner_job, minio):
    job_id, job_meta = bad_runner_job
    shepherd.enqueue_job(job_id, job_meta)  # runner should not start (and health-check should discover it)
    gevent.sleep(2)
    assert not shepherd['bare_sheep'].running
    assert shepherd.is_job_done(job_id)
    assert minio_object_exists(minio, job_id, 'error')
