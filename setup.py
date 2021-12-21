from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import sys
import numpy
import os
import os.path as path

force = False
profile = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

compilation_includes = [".", numpy.get_include()]

setup_path = path.dirname(path.abspath(__file__))

# build extension list
extensions = []
for root, dirs, files in os.walk(setup_path):
    for file in files:
        if path.splitext(file)[1] == ".pyx":
            pyx_file = path.relpath(path.join(root, file), setup_path)
            module = path.splitext(pyx_file)[0].replace("/", ".")
            extensions.append(Extension(module, [pyx_file], include_dirs=compilation_includes),)

cython_directives = {"language_level": 3}
if profile:
    cython_directives["profile"] = True


with open("README.md") as f:
    long_description = f.read()

setup(
    name="cherab-solps",
    version="1.2.0rc1",
    license="EUPL 1.1",
    namespace_packages=['cherab'],
    description="Cherab spectroscopy framework: SOLPS submodule",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/cherab",
    project_urls=dict(
        Tracker="https://github.com/cherab/solps/issues",
        Documentation="https://cherab.github.io/documentation/",
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["raysect==0.7.1", "cherab==1.3"],
    ext_modules=cythonize(extensions, force=force, compiler_directives=cython_directives),
)
