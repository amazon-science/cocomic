# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import logging

from tree_sitter import Language
from tree_sitter.binding import Parser

from cc_extractor.transform.python.context import PythonCodeContext
import timeout_decorator

logger = logging.getLogger(__name__)

DEFAULT_LIB_PATH = "build/python-lang-parser.so"

def load_parser(lang: str, lib_path: str = DEFAULT_LIB_PATH):
    language = Language(lib_path, lang)
    parser = Parser()
    parser.set_language(language)
    return parser


def get_file_content(file_path):
    contents = "".join(open(file_path).readlines())
    return contents

@timeout_decorator.timeout(10)
def get_parse_context(parser, contents, lang):
    code_bytes = bytes(contents, "utf8")
    tree = parser.parse(code_bytes)

    parsed_context = PythonCodeContext(tree,
                                        contents)

    return parsed_context
