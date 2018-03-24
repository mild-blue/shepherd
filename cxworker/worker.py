import gevent
from gevent.wsgi import WSGIServer
import zmq.green as zmq
from minio import Minio
import logging
import sys
from urllib3.exceptions import MaxRetryError

from .api.views import create_worker_blueprint
from .api import create_app
from .manager.registry import ContainerRegistry
from .manager.config import load_config, WorkerConfig
from .manager.output_listener import OutputListener


class Worker:
    def __init__(self):
        self.zmq_context = zmq.Context()
        self.app = create_app(__name__)
        self.registry = None
        self.minio = None
        self.config: WorkerConfig = None

    def load_config(self, config_stream):
        self.config = load_config(config_stream)
        logging.basicConfig(level=self.config.logging.log_level)
        self.registry = ContainerRegistry(self.zmq_context, self.config.registry, self.config.containers)
        self.minio = Minio(self.config.storage.url, self.config.storage.access_key, self.config.storage.secret_key,
                           self.config.storage.secure)

    def run(self, host: str, port: int):
        if self.config is None:
            raise RuntimeError("Configuration has not been loaded yet")

        self.app.register_blueprint(create_worker_blueprint(self.registry, self.minio))

        api_server = WSGIServer((host, port), self.app)
        output_listener = OutputListener(self.registry, self.minio)

        api = gevent.spawn(api_server.start)
        output = gevent.spawn(output_listener.listen)

        # everything should be ready, lets check if minio works
        try:
            self.minio.list_buckets()
            logging.info('Minio storage appears to be up and running')
        except MaxRetryError:
            logging.error('Could not list minio buckets. Verify minio url ({}) and credentials'
                          .format(self.config.storage.url))
            sys.exit(1)

        try:
            logging.info('Listening for API calls and worker outputs (send a keyboard interrupt to stop the worker)')
            gevent.joinall([api, output])
        except KeyboardInterrupt:
            self.registry.kill_all()
