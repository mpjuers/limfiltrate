#!/usr/bin/env python3
# Copyright 2022 Neko Juers

from distutils.core import setup

setup(
    name="LimFiltrate",
    version="0.1",
    description="A PCA-based visual tool for optimizing flow cytometry filters.",
    license="BSD",
    author="Neko Juers",
    author_email="neko.juers@gmail.com",
    url="https://github.com/mpjuers/limfiltrate",
    packages=["limfiltrate"],
    setup_requires=["black"]
)
