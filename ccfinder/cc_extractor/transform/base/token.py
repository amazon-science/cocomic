# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from collections import namedtuple

from tree_sitter.binding import Node as TSNode


class Token:
    """
    Wrapper class around tree-sitter Node
    """

    def __init__(self,
                 node: TSNode,
                 code_bytes: bytes):
        self.node = node
        self.code_bytes = code_bytes

    @property
    def extent(self):
        return self.code_bytes[self.node.start_byte:
                               self.node.end_byte].decode("utf8")

    @property
    def line_start(self):
        return self.node.start_point[1]

    @property
    def line_end(self):
        return self.node.end_point[1]

    def __str__(self):
        return f"Token('{self.extent}')"


class SplitPoint:
    """
    Represents a split point.
    The split point is represented by a Token + in token offsets
    """

    def __init__(self,
                 token: Token,
                 offset: int = 0):
        self.token = token
        self.offset = offset

    @property
    def token_line_start(self):
        return self.token.node.start_point[1]

    @property
    def token_line_end(self):
        return self.token.node.end_point[1]

    @property
    def token_partial_pos(self):
        return self.token.node.start_point[1] + min(self.offset, len(self.token.extent))


CodeLineTokens = namedtuple("CodeLineToken", ['line_no', 'tokens', 'text'])
