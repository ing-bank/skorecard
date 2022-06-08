from setuptools import setup, find_packages

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

base_packages = [
    "scipy>=1.5.2",
    "numpy>=1.19.5",
    "pandas>=1.1.5",
    "scikit-learn>=0.23.2",
    "dataclasses",
    "pyyaml",
    "category_encoders>=2.2.2",
]

dashboard_dep = [
    "dash>=1.21.0",
    "jupyter-dash>=0.4.0",
    "dash_bootstrap_components>=0.13",
]

reporting_dep = ["plotly>=4.14.3"]

dev_dep = [
    "flake8>=3.8.3",
    "black>=19.10b0",
    "pre-commit>=2.5.0",
    "mypy>=0.770",
    "flake8-docstrings>=1.4.0",
    "pytest>=6.0.0",
    "pytest-cov>=2.10.0",
]

docs_dep = [
    "mkdocs>=1.1.2",
    "mkdocs-material>=7.1",
    "mkdocstrings>=0.13.2",
    "mknotebooks>=0.7.0",
    "mkdocs-git-revision-date-localized-plugin>=0.7.2",
]

# Packages that are not a set together
# We recommend users to just install that package when it is used
# We use optbinning 0.8.0 or later, because later versions drop support for python 3.6
utils_dep = ["optbinning>=0.8.0"]

setup(
    name="skorecard",
    version="1.6.4",
    description="Tools for building scorecard models in python, with a sklearn-compatible API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ING Bank",
    author_email="daniel.timbrell@ing.com",
    license="MIT license",
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    include_package_data=True,
    install_requires=base_packages,
    extras_require={
        "base": base_packages,
        "dashboard": dashboard_dep,
        "reporting": reporting_dep,
        "all": base_packages + dashboard_dep + reporting_dep + dev_dep + docs_dep + utils_dep,
    },
    url="https://github.com/ing-bank/skorecard/",
    packages=find_packages(".", exclude=["tests", "notebooks", "docs"]),
)
