from distutils.core import setup
setup(
    name='nwn',
    version='1.0',
    author='Thomas Newton',
    py_modules=['nwn',],
    packages=['nwn',],
    install_required=[
        "numpy",
        "matplotlib",
        "networkx",
    ],
)