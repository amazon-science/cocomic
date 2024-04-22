# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from typing import List, Optional

from tree_sitter.binding import Node as TSNode

from cc_extractor.transform.base.base_context import BaseCodeContext


class BaseFunction:

    def __init__(self,
                 node: TSNode,
                 context: BaseCodeContext):
        self.node = node
        self.context = context

    @property
    def start_line_no(self):
        return self.node.start_point[0]

    @property
    def end_line_no(self):
        return self.node.end_point[0]

    @property
    def name(self) -> Optional[str]:
        """
        Return the function name in string, can be None if anonymous
        """
        raise NotImplemented

    @property
    def body(self) -> List[str]:
        """
        The actual body minus the documentation from the function
        """
        raise NotImplemented

    @property
    def block(self) -> TSNode:
        """
        The block, the node that contains all body statements
        """
        raise NotImplemented

    @property
    def num_preceding_chars(self) -> int:
        """
        Number for characters before this function definition
        """
        num_preceding_chars = 0
        for line_no, line_text in self.context.lines:
            if line_no < self.start_line_no:
                num_preceding_chars += len(line_text)
        return num_preceding_chars

    @property
    def empty(self) -> bool:
        """
        Check whether function body is empty or not
        """
        if self.block is None:
            return True
        return False

    @property
    def num_body_lines(self):
        """
        Number of lines in the function body
        """
        return len(self.body)

    @property
    def func_type(self):
        """
        Type of the function
        """
        return self.node.type

    @property
    def text(self):
        return self.context.get_node_text(self.node)

    @staticmethod
    def find_all(context: BaseCodeContext) -> List['BaseFunction']:
        """
        Given context retrieve all internal functions
        """
        raise NotImplemented

    def search_function_call(self, call_id: str) -> bool:
        """
        Search whether another function/callable is called within the function body
        """
        raise NotImplemented

    def get_docstring(self, **kwargs) -> Optional[TSNode]:
        """
        Check whether get_docstring is presented for the function or not
        """
        raise NotImplemented
