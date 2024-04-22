# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from typing import List, Callable

from cc_extractor.transform.base.base_context import BaseCodeContext
from tree_sitter.binding import Node as TSNode


class CodeBlock:
    """
    Wrapper class, including text and line numbers
    """

    def __init__(self, text: str,
                 start_lineno: int,
                 end_lineno: int):
        self.text = text
        self.start_lineno = start_lineno
        self.end_lineno = end_lineno

    def to_dict(self):
        return {
            "text": self.text,
            "line_no": [self.start_lineno, self.end_lineno + 1]
        }


class BaseFileContext:
    """
    Extract file level focal context, including
    1. Imports
    2. Function Level Context
    3. Class Level Context
    """

    def __init__(self, context: BaseCodeContext):
        # coding context
        self.context = context

    def imports(self):
        """
        Get all the imports from the file
        """
        raise NotImplemented

    def function_defs(self):
        """
        Get all function level definition from the file
        """
        raise NotImplemented

    def class_defs(self):
        """
        Get all class level definition from the file
        """
        raise NotImplemented

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

        self._dfs(self.context.tree.root_node, node_types, _cb)

        return result

    def get_code_block(self, node: TSNode) -> CodeBlock:
        """
        convert node to block
        """
        node_lines = self.context.get_node_window(node)
        st_lineno = node_lines[0].line_no
        ed_lineno = node_lines[-1].line_no

        return CodeBlock("\n".join([nl.text for nl in node_lines]),
                         st_lineno, ed_lineno)

    def parse(self):
        """
        Return consolidated file context including, imports/function definitions/class definitions
        """
        file_context = dict()
        file_context.update(self.imports())
        file_context.update(self.function_defs())
        file_context.update(self.class_defs())
        return file_context
