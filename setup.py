from setuptools import setup, find_packages

tests_require = [
    'pytest',
    'pytest-mock',
    'pytest-aiohttp',
    'molotov',
]

setup(name='shepherd',
      version='0.5.3',
      description='Shepherd',
      long_description='Asynchronous worker',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7'
      ],
      keywords='worker',
      url='https://github.com/iterait/shepherd',
      author=['Iterait a.s.', 'Cognexa Solutions s.r.o.'],
      author_email='info@iterait.com',
      license='MIT',
      packages=['shepherd']+['.'.join(('shepherd', package)) for package in find_packages('shepherd')],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['pytest-runner'],
      tests_require=tests_require,
      install_requires=[
        'click==7.1.2',
        'simplejson==3.17.6',
        'pyzmq==22.3.0',
        'ruamel.yaml==0.17.17',
        'requests==2.26.0',
        'schematics==2.1.1',
        'aiohttp==3.7.4',
        'aiohttp-cors==0.7.0',
        'apistrap==0.9.11',
        'minio==5.0.6',
        'urllib3==1.24.2'
      ],
      extras_require={
          'docs': ['sphinx>=2.0', 'autoapi>=1.4', 'sphinx-argparse',
                   'sphinx-autodoc-typehints', 'sphinx-bootstrap-theme'],
          'tests': tests_require,
      },
      entry_points={
          'console_scripts': [
              'shepherd=shepherd.manage:run',
              'shepherd-runner=shepherd.runner.runner_entry_point:main'
          ]
      }
)
