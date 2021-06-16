<img src="https://github.com/ing-bank/skorecard/raw/main/docs/assets/img/skorecard_logo.svg" width="150" align="right">

# skorecard

<!-- ![pytest](https://github.com/ing-bank/skorecard/workflows/Release/badge.svg) -->
![pytest](https://github.com/ing-bank/skorecard/workflows/Development/badge.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/skorecard)
![PyPI](https://img.shields.io/pypi/v/skorecard)
![PyPI - License](https://img.shields.io/pypi/l/skorecard)
![GitHub contributors](https://img.shields.io/github/contributors/ing-bank/skorecard)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

`skorecard` is a scikit-learn compatible python package that helps streamline the development of credit risk acceptance models (scorecards).

Scorecards are ‘traditional’ models used by banks in the credit decision process. Internally, scorecards are Logistic Regressions models that make use of features that are binned into different groups. The process of binning is usually done manually by experts, and `skorecard` provides tools to makes this process easier. `skorecard` is built on top of [scikit-learn](https://pypi.org/project/scikit-learn/) as well as other excellent open source projects like [optbinning](https://pypi.org/project/optbinning/), [dash](https://pypi.org/project/dash/) and [plotly](https://pypi.org/project/plotly/).

## Features ⭐

- Automate bucketing of features inside scikit-learn pipelines.
- Dash webapp to help manually tweak bucketing of features with business knowledge (*not yet available*)
- Extension to `sklearn.linear_model.LogisticRegression` that is also able to report p-values
- Plots and reports to speed up analysis and writing technical documentation.

## Installation

```shell
pip3 install skorecard
```

## Documentation

See [ing-bank.github.io/skorecard/](https://ing-bank.github.io/skorecard/).

## Presentations

| Title                                              | Host                    | Date         | Speaker(s)                                   |
|----------------------------------------------------|-------------------------|--------------|----------------------------------------------|
| Skorecard: Making logistic regressions great again | [ING Data Science Meetup](https://www.youtube.com/watch?v=UR_1XZxEuCw) | 10 June 2021 | Daniel Timbrell, Sandro Bjelogrlic, Tim Vink |
