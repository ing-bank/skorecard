"""
Tests all the codeblocks in the docstrings, making sure they execute.

This approach is adapted from, and explained in:
https://calmcode.io/docs/epic.html

Try it out with:

```bash
pytest tests/test_docstring.py --verbose
```
"""

import pytest

from tests.conftest import CLASSES_TO_TEST, FUNCTIONS_TO_TEST


def get_public_methods(cls_ref):
    """Helper test function, gets all public methods in a class."""
    return [m for m in dir(cls_ref) if m == "__init__" or not m.startswith("_")]


def get_test_pairs(classes_to_test):
    """Helper test function, get tuples with class and public method."""
    test_pairs = []
    for cls_ref in classes_to_test:
        for meth_ref in get_public_methods(cls_ref):
            test_pairs.append((cls_ref, meth_ref))
    return test_pairs


def handle_docstring(doc, indent):
    """
    Check python code in docstring.

    This function will read through the docstring and grab
    the first python code block. It will try to execute it.
    If it fails, the calling test should raise a flag.
    """
    if not doc:
        return
    start = doc.find("```python\n")
    end = doc.find("```\n")
    if start != -1:
        if end != -1:
            code_part = doc[(start + 10) : end].replace(" " * indent, "")
            print(code_part)
            exec(code_part)


# @pytest.mark.parametrize("m", MODULES_TO_TEST)
# def test_module_docstrings(m):
#     """
#     Take the docstring of a given module.

#     The test passes if the usage examples causes no errors.
#     """
#     handle_docstring(m.__doc__, indent=0)


@pytest.mark.parametrize("c", CLASSES_TO_TEST)
def test_class_docstrings(c):
    """
    Take the docstring of a given class.

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(c.__doc__, indent=4)


@pytest.mark.parametrize("clf_ref,meth_ref", get_test_pairs(CLASSES_TO_TEST))
def test_method_docstrings(clf_ref, meth_ref):
    """
    Take the docstring of every method (m) on the class (c).

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(getattr(clf_ref, meth_ref).__doc__, indent=8)


@pytest.mark.parametrize("f", FUNCTIONS_TO_TEST)
def test_function_docstrings(f):
    """
    Take the docstring of every function.

    The test passes if the usage examples causes no errors.
    """
    handle_docstring(f.__doc__, indent=4)
