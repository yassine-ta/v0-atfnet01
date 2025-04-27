#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="atfnet",
    version="0.1.0",
    author="ATFNet Team",
    author_email="info@atfnet.com",
    description="Adaptive Time-Frequency Network for Trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atfnet/atfnet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
        "tensorflow>=2.3.0",
        "PyQt5>=5.15.0",
        "pyyaml>=5.3.0",
        "MetaTrader5>=5.0.29",
        "keras",
        "pyqtgraph",
        "PyQtWebEngine",
        "mplfinance",
    ],
    entry_points={
        "console_scripts": [
            "atfnet=atfnet.run_atfnet:main",
            "atfnet-backtest=atfnet.backtest.run_backtest:main",
            "atfnet-optimize=atfnet.backtest.run_optimization:main",
            "atfnet-ui=atfnet.app.run_enhanced_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "atfnet": ["config/*.yaml", "config/*.yml"],
    },
)
