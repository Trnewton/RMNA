from distutils.core import setup
setup(
    name='rmna',
    version='1.0',
    author='Thomas Newton',
    py_modules=['rmna','graph','display'],
    install_required=[
        "numpy",
        "matplotlib",
        "networkx",
    ],
)