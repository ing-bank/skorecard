# Contributing guidelines

Make sure to discuss any changes you would like to make in the issue board, before putting in any work.

## Setup

Development install:

```shell
pip install -e '.[all]'
```

Unit testing:

```shell
pytest
```

We use [pre-commit](https://pre-commit.com/) hooks to ensure code styling. Install with:

```shell
pre-commit install
```

## Documentation

We use [mkdocs](https://www.mkdocs.org) with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/) theme. The docs are structured using the [divio documentation system](https://documentation.divio.com/). To view the docs locally:

```shell
pip install mkdocs-material
mkdocs serve
```

## Releases and versioning

We use [semver](https://semver.org/) for versioning. When we are ready for a release, the maintainer runs:

```shell
git tag -a v0.1 -m "skorecard v0.1" && git push origin v0.1
```

When we create a new github release a [github action](https://github.com/ing-bank/skorecard/blob/main/.github/workflows/publish_pypi.yml) is triggered that:

- a new version will be deployed to pypi
- the docs will be re-built and deployed

## Logo

- We adapted the ['scores' noun](https://thenounproject.com/search/?q=score&i=1929515)
- We used [this color scheme](https://coolors.co/d7263d-f46036-2e294e-1b998b-c5d86d) from coolors.co 
- We edited the logo using https://boxy-svg.com/app

## Terminology

- `BucketMapping` is a custom class that stores all the information needed for bucketing, including the map itself (either boundaries for binning, or a list of lists for categoricals)
- `FeaturesBucketMapping` is simply a collection of `BucketMapping`s, and is used to store all info for bucketing transformations for a dataset.

