# !/usr/bin/env python3
# -*- coding:utf-8 -*-

import csv
import json
import os
import pickle as pk
import sys

import yaml

__all__ = [
    "load_yaml",
    "dump_yaml",
    "load_pickle",
    "dump_pickle",
    "load_json",
    "dump_json",
    "load_csv",
    "dump_csv",
    "PK_HIGHEST_PROTOCOL",
    "PK_DEFAULT_PROTOCOL",
    "PK_PROTOCOL_0",
    "PK_PROTOCOL_1",
    "PK_PROTOCOL_2",
    "PK_PROTOCOL_3",
    "PK_PROTOCOL_4",
]

PK_HIGHEST_PROTOCOL = pk.HIGHEST_PROTOCOL
PK_DEFAULT_PROTOCOL = pk.DEFAULT_PROTOCOL

PK_PROTOCOL_0 = 0
PK_PROTOCOL_1 = 1
PK_PROTOCOL_2 = 2
PK_PROTOCOL_3 = 3
PK_PROTOCOL_4 = 4


def load_yaml(filename, method="r", encoding="utf-8"):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.
    """
    if "b" in method:
        with open(filename, method) as f:
            return yaml.load(stream=f, Loader=yaml.FullLoader)
    else:
        with open(filename, method, encoding=encoding) as f:
            return yaml.load(stream=f, Loader=yaml.FullLoader)


def dump_yaml(data, filename, method="w", encoding="utf-8", safe_mode=False, **kwargs):
    """
    Serialize a Python object into a YAML stream.
    If stream is None, return the produced string instead.
    """
    if safe_mode:
        temp_file_name = filename + "_temp"
    else:
        temp_file_name = filename

    with open(temp_file_name, method) as f:
        if "b" in method:
            ret = yaml.dump(data=data, stream=f, encoding=encoding, **kwargs)
        else:
            ret = yaml.dump(data=data, stream=f, **kwargs)
        if safe_mode:
            f.flush()
            os.fsync(f.fileno())

    if safe_mode:
        os.rename(temp_file_name, filename)

    return ret


def load_pickle(filename, method="rb"):
    """
    Read a pickled object representation from the open file.
    Return the reconstituted object hierarchy specified in the file.
    """
    if method == "r":
        method = "rb"
        sys.stderr.write("loadPk: using 'rb' instead of 'r'\n")
    with open(filename, method) as f:
        return pk.load(f)


def dump_pickle(
    data, filename, method="wb", protocol=PK_DEFAULT_PROTOCOL, safe_mode=False
):
    """
    Write a pickled representation of obj to the open file.
    """
    if method == "w":
        method = "wb"
        sys.stderr.write("dumpPk: using 'wb' instead of 'w'\n")

    if safe_mode:
        temp_file_name = filename + "_temp"
    else:
        temp_file_name = filename

    with open(temp_file_name, method) as f:
        pk.dump(obj=data, file=f, protocol=protocol)

        if safe_mode:
            f.flush()
            os.fsync(f.fileno())

    if safe_mode:
        os.rename(temp_file_name, filename)


def load_json(filename, method="r", **kwargs):
    with open(filename, method) as f:
        data = json.load(f, **kwargs)

    return data


def dump_json(data, filename, method="w", indent=None, safe_mode=False, **kwargs):
    """
    Dump data to json file

    Args:
        data: json serializable python object, dict, list etc...
            dumped data
        filename: str or Path
            path of dumped json file
        method(str): an optional string that specifies the mode in which the file
            is opened
        indent: None or int
            If ``indent`` is a non-negative integer, then JSON array elements and
            object members will be pretty-printed with that indent level. An indent
            level of 0 will only insert newlines. ``None`` is the most compact
            representation.
        safe_mode(bool): Dump in safe mode or not
    """
    if safe_mode:
        temp_file_name = filename + "_temp"
    else:
        temp_file_name = filename

    if "b" in method:
        raise ValueError("can not dump json with binary mode!")

    with open(temp_file_name, method) as f:
        json.dump(obj=data, fp=f, indent=indent, **kwargs)

        if safe_mode:
            f.flush()
            os.fsync(f.fileno())

    if safe_mode:
        os.rename(temp_file_name, filename)


def load_csv(filename, method="r"):
    """Load csv from file

    Args:
        filename: filename of csv file
        method(str): an optional string that specifies the mode in which the file
            is opened

    Returns:
        List: a list of line contents, an empty list will be returned if csv file is empty
    """
    with open(filename, method, newline="") as f:
        reader = csv.reader(f)
        lines = [line for line in reader]

    return lines


def dump_csv(data, filename, method="w", safe_mode=False):
    """Dump given data to a cvs file

    Args:
        data:
        filename:
        method:
        safe_mode:

    Returns:

    """
    if not isinstance(data, (list, tuple)):
        data = [data]

    if "b" in method:
        method = "w"

    if safe_mode:
        temp_file_name = filename + "_temp"
    else:
        temp_file_name = filename

    with open(temp_file_name, method) as f:
        writer = csv.writer(f)
        for item in data:
            writer.writerow(item)

        if safe_mode:
            f.flush()
            os.fsync(f.fileno())

    if safe_mode:
        os.rename(temp_file_name, filename)
