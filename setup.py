from setuptools import setup
from setuptools import find_packages

# get the __version__ variable
exec(open("./iuml/version.py").read())

setup(name = 'iUNU Object Detection',
    version = __version__,
    description = 'Bud detection training utilities',
    author = 'iUNU',
    install_requires=['numpy>=1.9.1',
                       'keras>=2.1.3',
                        'scipy>=0.14',
                        'randomcolor',
                        'pydensecrf',
                        'requests',
                        'keras-resnet',
                     ],

    packages = find_packages()
)
