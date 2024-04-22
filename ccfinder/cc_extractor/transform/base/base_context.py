# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import logging
from collections import defaultdict
from typing import List, Dict, Callable

from tree_sitter.binding import Node as TSNode
from tree_sitter.binding import Tree as TSTree

from cc_extractor.transform.base.token import (Token,
                                CodeLineTokens)

logger = logging.getLogger(__name__)


class BaseCodeContext:
    """
    Object to hold various parsed code structure, like
    lines of code, tokens stream, etc
    """

    def __init__(self,
                 tree: TSTree,
                 code: str,
                 lang: str):
        self.tree = tree  # parsed AST tree
        self.code = code
        self.lang = lang
        self.code_bytes = bytes(self.code, "utf8")  # for validation mainly
        self.parsed_lines = []  # will be filled by implementation later

    @staticmethod
    def validate_token(node: TSNode,
                       line_text: str,
                       code_bytes: bytes):
        """
        Validate the token's integrity
        """
        start_line_no, line_start = node.start_point
        end_line_no, line_end = node.end_point
        # offset in the
        byte_start = node.start_byte
        byte_end = node.end_byte

        line_span_text = bytes(line_text[line_start: line_end], encoding="utf8")
        bytes_span_text = code_bytes[byte_start: byte_end]
        if line_span_text != bytes_span_text:
            if start_line_no == end_line_no:
                # not a multiline node
                logger.debug(f"Node: {node}'s byte span {repr(bytes_span_text)} "
                             f"doesn't match its line span {repr(line_span_text)}, line{line_text}")

    @staticmethod
    def is_multiline_token(token: Token):
        start_line_no = token.node.start_point[0]
        end_line_no = token.node.end_point[0]
        if start_line_no != end_line_no:
            # multi-line string
            return True
        return False

    def assign_token_to_line(self, node: TSNode, line_tokens: Dict[int, List]):
        """
        Implement language specific assignment strategy
        """
        raise NotImplemented

    def _parse(self) -> List[CodeLineTokens]:
        line_tokens = defaultdict(list)
        line_texts = self.code.split("\n")

        # walk through the tree to align all tokens
        # to corresponding lines
        self.assign_token_to_line(self.tree.root_node,
                                  line_tokens)

        parsed_line_tokens = list()
        # print(f"line_texts:{line_texts}, line_tokens: {line_tokens}")
        # for lines that are not collected
        # probably empty lines, insert them first
        for line_no, line_text in enumerate(line_texts):
            if line_no not in line_tokens:
                logger.debug(f"Line {line_no} not collected {line_text}")
                line_tokens[line_no] = []

        sorted_line_tokens = sorted(line_tokens.items(), key=lambda i: i[0])

        for line_no, token_nodes in sorted_line_tokens:
            text = line_texts[line_no]
            tokens = list()
            for node in token_nodes:
                # valid each token to make sure its node start/end points
                # are consistent with its byte location
                self.validate_token(node, text, self.code_bytes)
                tokens.append(Token(node, self.code_bytes))
            parsed_line_tokens.append(CodeLineTokens(line_no=line_no,
                                                     tokens=tokens,
                                                     text=text))

        # print(parsed_line_tokens)
        assert len(parsed_line_tokens) == len(line_texts)
        return parsed_line_tokens

    @staticmethod
    def iter_token(tokenized_code_lines: List[CodeLineTokens]):
        prev = None
        for clt in tokenized_code_lines:
            for token in clt.tokens:
                if prev is not None and prev.node != token.node:
                    # in case multiple line share the same token
                    yield token
                prev = token

    @property
    def tokens(self):
        """
        Retrieve all tokens in a stream
        """
        return self.iter_token(self.parsed_lines)

    def tokens_from_line_no(self, line_no):
        """
        Retrieve tokens from a certain line
        """
        return self.iter_token(self.parsed_lines[line_no:])

    @property
    def lines(self):
        """
        Retrieve all lines
        """
        for clt in self.parsed_lines:
            yield (clt.line_no, clt.text)

    @property
    def num_of_lines(self) -> int:
        return len(self.parsed_lines)

    @property
    def syntax_error(self) -> bool:
        """
        Check whether this is syntax error in the AST
        """
        try:
            def dfs_check(node):
                """
                Check whether this is node that can't be parsed
                """
                if node.type == "ERROR":
                    return True
                for c in node.children:
                    if dfs_check(c):
                        return True
                return False

            return dfs_check(self.tree.root_node)
        except:
            # some js cases will exceed maximum recursion depth
            return True

    @property
    def max_line_length(self) -> int:
        """
        The max line length among all lines,
        without parsing
        """
        lengths = [len(line) for line in self.code.split("\n")]
        return max(lengths)

    def get_token_line_no(self, token: Token) -> int:
        """
        Given on token, retrieve which line it is in
        """
        node = token.node
        st_line_no = node.start_point[0]

        return st_line_no

    def get_line(self, line_no) -> CodeLineTokens:
        """
        Get single line given line_no
        """
        code_line = self.parsed_lines[line_no]
        assert code_line.line_no == line_no
        return code_line

    def get_window_by_line(self, start_line_no, end_line_no) -> List[CodeLineTokens]:
        return self.parsed_lines[start_line_no:
                                 end_line_no]

    def get_node_text(self, node: TSNode):
        """
        Helper function to extract node text content
        """
        node_start_byte = node.start_byte
        node_end_byte = node.end_byte
        return self.code_bytes[node_start_byte:
                               node_end_byte].decode("utf8")

    def get_node_window(self, node: TSNode):
        node_start_lineno = node.start_point[0]
        node_end_lineno = node.end_point[0]

        return self.parsed_lines[node_start_lineno: node_end_lineno + 1]

    def find_last_code_lines(self, num_lines: int) -> List[CodeLineTokens]:
        """
        Fina last N valid code lines
        """
        reverse_code_liens = self.parsed_lines[:]
        reverse_code_liens.reverse()

        valid_code_lines = []

        for cl in reverse_code_liens:
            # check the line is not empty
            # and can be just comment or string
            if len(valid_code_lines) == num_lines:
                break
            if self.is_code_line(cl.line_no):
                valid_code_lines.append(cl)

        valid_code_lines.reverse()
        return valid_code_lines

    def is_code_line(self, line_no) -> bool:
        """
        Check whether a given line is code or not
        """
        raise NotImplemented

    def check_task_exists_in_code(self, prompt: str, groundtruth: str):
        """
        Check the given prompt + groundtruth, can be found in original
        code, meaning the code itself is not modified
        """
        if self.code.find(prompt + groundtruth) == -1:
            raise ValueError("Task Instance are not in file")
        return True

    def _dfs(self,
             node: TSNode,
             node_types: List[str],
             callback: Callable):
        """
        Helper to traverse parsed AST
        """
        if node.type in node_types:
            callback(node)

        for child in node.children:
            self._dfs(child, node_types, callback)

    def collect_nodes(self, node_types: List[str]) -> List[TSNode]:
        """
        Collect all nodes that belong to certain types
        """
        result = list()

        def _cb(n):
            result.append(n)

        self._dfs(self.tree.root_node, node_types, _cb)

        return result
