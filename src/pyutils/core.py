#!/usr/bin/env python
# -*- coding: utf-8 -*-

import enum
import re
import shutil
from pathlib import Path
from typing import Iterator, List, Union

__all__ = [
    "find_numbers",
    "glob_dir",
    "copy_empty_tree",
    "make_empty_dir",
    "check_isinstance",
    "check_in_list",
    "check_attr_in_list",
    "check_getitem",
    "EnhanceEnum",
]


def find_numbers(string: str) -> List[int]:
    """Find all numbers(integers) in a string and return a list of found numbers

    Args:
        string(str): source string

    Returns:
        numbers(list): list of found numbers, an empty list will be returned if
        no numbers is found.

    Examples:
    >>> s = "abc123gf54"
    [123, 54]
    >>> s = "abcdeg"
    []
    """

    numbers = [int(num_str) for num_str in re.findall(r"\d+", string)]

    return numbers


def glob_dir(
    dir_,
    include_patterns: Union[List[str], None] = None,
    exclude_patterns: Union[List[str], None] = None,
    ignore_case=False,
) -> Iterator[Path]:
    """Glob directory recursively and filter paths by extensions and filename suffix

    Args:
        dir_: directory path
        include_patterns: list of include patterns, eg ["*.png", "*.txt"]
        exclude_patterns: list of exclude patterns, eg: ["*mask.png", "*mask.txt"]
        ignore_case: case sensitive match or not

    Returns:
        Iterator[Path]: an iterator of matched paths under given directory

    Raises:
        FileNotFoundError: If given directory dose not exist

    Examples:
    # glob current directory, get .py files without *version.py like files
    >>> glob_dir(".", include_patterns=["*.py"], exclude_patterns=["*version.py"])
    """
    dir_ = Path(dir_)

    if not dir_.is_dir():
        raise FileNotFoundError(f"directory {dir_} dose not exist!")

    def all_pass_filter(_):
        return True

    def _include_filter(p: Path):
        if ignore_case:
            p = Path(p.as_posix().lower())

        return any(p.match(pattern) for pattern in include_patterns)  # type: ignore

    def _exclude_filter(p: Path):
        if ignore_case:
            p = Path(p.as_posix().lower())

        return all(not p.match(pattern) for pattern in exclude_patterns)  # type: ignore

    include_filter = _include_filter if include_patterns else all_pass_filter
    exclude_filter = _exclude_filter if exclude_patterns else all_pass_filter

    def path_filter(path: Path):
        return include_filter(path) and exclude_filter(path)

    return filter(path_filter, dir_.rglob("*"))


def copy_empty_tree(src, dst, exist_ok: bool = False) -> None:
    """Copy the folder structure of source directory to destination directory,
    make src and dst have the same folder structure.

    Args:
        src: source directory path
        dst: destination directory path
        exist_ok(bool): If dst is already exist, dst will be emptied first when exist_ok is True,
                        and raise a error while exist_ok is False.

    Examples:
        src
          |-- sub1
                |-- sub11
                        |-- file111
                        |-- file112
                |-- sub12
          |-- sub2
                |--sub21
        dst
          |-- ...

    >>> copy_empty_tree(src, dst)
        dst
          |-- sub1
                |-- sub11
                |-- sub13
          |-- sub2
                |--sub21
    Raises:
        FileNotFoundError: If src dose not exist or is not a directory
        ValueError: If src and dst is the same directory

    """
    src = Path(src)
    dst = Path(dst)

    if not src.is_dir():
        raise FileNotFoundError("{} is not a directory.".format(str(src)))
    if src == dst:
        raise ValueError("src and dst could not be the same.")

    def ignore(dir_, contents):
        dir_ = Path(dir_)

        ignored_contents = [
            content for content in contents if not (dir_ / content).is_dir()
        ]

        return ignored_contents

    if dst.is_dir():
        if exist_ok:
            shutil.rmtree(dst)
        else:
            raise RuntimeError(
                "'{}' already exist. Use an nonexistent directory as destination "
                "or set exist_ok=True".format(dst)
            )

    shutil.copytree(src, dst, ignore=ignore)


def make_empty_dir(dir_, exist_ok: bool = False, parents: bool = False):
    """If dir_ exists, reset the directory, otherwise, make a new directory.

    Args:
        dir_: directory path
        exist_ok(bool): Flags of error raised when the target directory already exists. Refer to pathlib.
        parents(bool): Flags of weather missing parents of the target directory would be created.
                        Refer to pathlib.
    """

    dir_ = Path(dir_)
    if dir_.exists() and (not exist_ok):
        raise RuntimeError(
            "'{}' already exist. Use an nonexistent directory path "
            "or set exist_ok=True".format(dir_)
        )

    shutil.rmtree(dir_, ignore_errors=True)
    dir_.mkdir(parents=parents, exist_ok=True)


def check_isinstance(_types, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is an instance
    of one of *_types*; if not, raise an appropriate TypeError.

    As a special case, a ``None`` entry in *_types* is treated as NoneType.

    Examples:
    >>> check_isinstance((SomeClass, None), arg=arg)
    """
    types = _types
    none_type = type(None)
    types = (
        (types,)
        if isinstance(types, type)
        else (none_type,)
        if types is None
        else tuple(none_type if tp is None else tp for tp in types)
    )

    def type_name(tp):
        return (
            "None"
            if tp is none_type
            else tp.__qualname__
            if tp.__module__ == "builtins"
            else f"{tp.__module__}.{tp.__qualname__}"
        )

    for k, v in kwargs.items():
        if not isinstance(v, types):
            names = [*map(type_name, types)]
            if "None" in names:  # Move it to the end for better wording.
                names.remove("None")
                names.append("None")
            raise TypeError(
                "{!r} must be an instance of {}, not a {}".format(
                    k,
                    ", ".join(names[:-1]) + " or " + names[-1]
                    if len(names) > 1
                    else names[0],
                    type_name(type(v)),
                )
            )


def check_in_list(_values, *, _print_supported_values=True, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is in *_values*.

    Args:
        _values : iterable
            Sequence of values to check on.
        _print_supported_values : bool, default: True
            Whether to print *_values* when raising ValueError.
        **kwargs : dict
            *key, value* pairs as keyword arguments to find in *_values*.

    Raises:
        ValueError: If any *value* in *kwargs* is not found in *_values*.

    Returns:
    >>> check_in_list(["foo", "bar"], arg=arg, other_arg=other_arg)
    """
    values = _values
    for key, val in kwargs.items():
        if val not in values:
            if _print_supported_values:
                raise ValueError(
                    f"{val!r} is not a valid value for {key}; "
                    f"supported values are {', '.join(map(repr, values))}"
                )
            else:
                raise ValueError(f"{val!r} is not a valid value for {key}")


def check_attr_in_list(
    attr_name: str, valid_values, *, _print_supported_values=True, **kwargs
):
    """

    For each *key, value* pair in *kwargs*, check the attribute *value*.*attr_name* is valid

    Args:
        attr_name(str): name of the attribute to be checked.
        valid_values: list
            Sequence of valid values of *attr_name*.
        _print_supported_values : bool, default: True
            Whether to print *_values* when raising ValueError.

        **kwargs : dict
            *key, value* pairs as keyword arguments to find in *_values*.

    Raises:
        ValueError: If the attributes of any *value* in *kwargs* checking failed

    Examples:
        >>> check_attr_in_list('arrti_name', valid_values=[v1], test_1=test_1, test_2=test_2)
        >>> check_attr_in_list('arrti_name', valid_values=[v1, v2], test_1=test_1, test_2=test_2)

    """

    item_value_strs = list(map(str, valid_values))
    if len(item_value_strs) > 1:
        valid_str = ", ".join(item_value_strs[:-1]) + " or " + item_value_strs[-1]
    else:
        valid_str = item_value_strs[0]

    for arg_k, arg_v in kwargs.items():
        attr_value = getattr(arg_v, attr_name)

        if attr_value not in valid_values:

            if _print_supported_values:
                raise ValueError(
                    f"expect attribute {attr_name} to be {valid_str}, "
                    f"while the {attr_name} of {arg_k!r} is {attr_value}"
                )

            else:
                raise ValueError(
                    f"attribute value {attr_value} of {arg_k!r} is not valid"
                )


def check_getitem(_mapping, **kwargs):
    """
    *kwargs* must consist of a single *key, value* pair.  If *key* is in
    *_mapping*, return ``_mapping[value]``; else, raise an appropriate
    ValueError.

    Examples:
    >>> check_getitem({"foo": "bar"}, arg="foo")
    'bar'
    """
    mapping = _mapping
    if len(kwargs) != 1:
        raise ValueError("check_getitem takes a single keyword argument")
    ((k, v),) = kwargs.items()
    try:
        return mapping[v]
    except KeyError:
        raise ValueError(
            "{!r} is not a valid value for {}; supported values are {}".format(
                v, k, ", ".join(map(repr, mapping))
            )
        ) from None


class EnhanceEnum(enum.Enum):
    @classmethod
    def member_values(cls) -> list:
        return [i.value for i in cls._member_map_.values()]  # type: ignore
