#!/usr/bin/env python3
#
#   Setup for FGb python interface
#
#   Based on the FGb Sage interface
#   https://github.com/mwageringel/fgb_sage
#

import os, requests, tarfile

from setuptools import setup, Command
from distutils.extension import Extension
from distutils.command import clean
from distutils.file_util import copy_file
from distutils.dir_util import copy_tree, mkpath, remove_tree
from Cython.Build import cythonize

DIR = os.path.dirname(__file__)
if DIR != "":
    DIR += os.path.sep

VERSION = 1
FGB_TAR_URL = "https://www-polsys.lip6.fr/~jcf/FGb/C/@downloads/call_FGb6.maclinux.x64.tar.gz"

class FetchFGb(Command):
    description = "download and extract libfgb"
    user_options = []

    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        print("Downloading fgb library")
        # TODO enable verification, but upstream has bad ssl
        resp = requests.get(FGB_TAR_URL, stream=True, verify=False)

        print("Extracting tarfile")
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as archive:
            archive.extractall(f"{DIR}local/fgb/")

        print("Unpacking")
        mkpath(f"{DIR}local/lib")
        copy_tree(f"{DIR}local/fgb/call_FGb/nv/maple/C/x64", f"{DIR}local/lib")
        mkpath(f"{DIR}local/include")
        copy_tree(f"{DIR}local/fgb/call_FGb/nv/protocol", f"{DIR}local/include")
        copy_file(f"{DIR}local/fgb/call_FGb/nv/maple/C/call_fgb.h", f"{DIR}local/include/call_fgb.h")
        copy_file(f"{DIR}local/fgb/call_FGb/nv/maple/C/call_fgb_basic.h", f"{DIR}local/include/call_fgb_basic.h")
        copy_file(f"{DIR}local/fgb/call_FGb/nv/int/protocol_maple.h", f"{DIR}local/include/protocol_maple.h")
        remove_tree(f"{DIR}local/fgb")

        dir_arg = ""
        if DIR != "":
            dir_arg = f"--directory={DIR}"

        os.system(f"patch {dir_arg} -p1 < {DIR}patches/fix_fgb_h.patch")

        print("Done")

class CleanCommand(clean.clean):
    def run(self):
        if os.path.exists(f"{DIR}build"):
            remove_tree(f"{DIR}build")
        clean.clean.run(self)

class DistCleanCommand(CleanCommand):
    def run(self):
        super().run()
        if os.path.exists(f"{DIR}local"):
            remove_tree(f"{DIR}local")

setup(
    name = "fgb_py",
    package_dir = "fgb",
    version = VERSION,
    cmdclass = {
        "build_libfgb" : FetchFGb,
        "clean"        : CleanCommand,
        "distclean"    : DistCleanCommand,
    },
    options = {
        "build" : {
            "build_lib"   : f"{DIR}build",
            "build_base"  : f"{DIR}build",
        },
    },
    ext_modules = cythonize([Extension(
        name = "fgb",
        include_dirs = [f"{DIR}local/include"],
        library_dirs = [f"{DIR}local/lib"],
        libraries = ["fgb", "fgbexp", "gb", "gbexp", "minpoly", "minpolyvgf", "gmp", "m", "stdc++"],
        extra_compile_args = ["-fopenmp"],
        extra_link_args = ["-fopenmp"],
        sources = [f"{DIR}src/fgb.pyx"]
    )], build_dir = f"{DIR}build")
)
