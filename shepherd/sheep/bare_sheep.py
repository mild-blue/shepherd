import os
import shlex
import subprocess
from typing import Dict, Any, Optional

from schematics.types import StringType

from .base_sheep import BaseSheep
from .docker_sheep import extract_gpu_number
from ..errors.sheep import SheepConfigurationError


class BareSheep(BaseSheep):
    """
    An adapter that running models on bare metal with ``shepherd-runner``.
    This might be useful when Docker isolation is impossible or not necessary, for example in deployments with just a
    few models.
    """

    class Config(BaseSheep.Config):
        working_directory: str = StringType(required=True)  # working directory of the shepherd-runner
        stdout_file: Optional[str] = StringType(required=False)  # if specified, capture runner's stdout to this file
        stderr_file: Optional[str] = StringType(required=False)  # if specified, capture runner's stderr to this file

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Create new :py:class:`BareSheep`.

        :param config: bare sheep configuration (:py:class:`BareSheep.Config`)
        :param kwargs: parent's kwargs
        """
        super().__init__(**kwargs)
        self._config: self.Config = self.Config(config)
        self._runner: Optional[subprocess.Popen] = None
        self._runner_config_path: Optional[str] = None

    def start(self, model_name: str, model_version: str) -> None:
        """
        Start a subprocess with the sheep runner.

        :param model_name: model name
        :param model_version: model version
        """
        super().start(model_name, model_version)

        # prepare env. variables for GPU computation and stdout/stderr files
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(filter(None, map(extract_gpu_number, self._config.devices)))
        stdout = subprocess.DEVNULL

        try:
            if self._config.stdout_file is not None:
                os.makedirs(os.path.dirname(self._config.stdout_file), exist_ok=True)
                stdout = open(self._config.stdout_file, 'a')
        except IOError as ex:
            raise SheepConfigurationError('Could not open stdout log file: {}'.format(str(ex))) from ex

        stderr = subprocess.DEVNULL

        try:
            if self._config.stderr_file is not None:
                os.makedirs(os.path.dirname(self._config.stderr_file), exist_ok=True)
                stderr = open(self._config.stderr_file, 'a')
        except IOError as ex:
            raise SheepConfigurationError('Could not open stderr log file: {}'.format(str(ex))) from ex

        # start the runner in a new sub-process
        self._runner = subprocess.Popen(
            shlex.split('shepherd-runner -p {} {}'.format(self._config.port, self._runner_config_path)), env=env,
            cwd=self._config.working_directory, stdout=stdout, stderr=stderr)

    def slaughter(self) -> None:
        """Kill the underlying runner (subprocess)."""
        super().slaughter()
        if self._runner is not None:
            self._runner.kill()
            self._runner = None

    @property
    def running(self) -> bool:
        """Check if the underlying runner (subprocess) is running."""
        return self._runner is not None and self._runner.poll() is None
