#!/usr/bin/env python
# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from tree_sitter import Language


def build_language_lib():
    Language.build_library(
        # Store the library in the `build` directory
        'ccfinder/cc_extractor/build/python-lang-parser.so',

        # Include one or more languages
        [
            'ts_package/tree-sitter-python'
        ]
    )


if __name__ == "__main__":
    build_language_lib()