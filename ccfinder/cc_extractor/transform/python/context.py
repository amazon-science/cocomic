import logging
from typing import List, Dict

from tree_sitter.binding import Node as TSNode
from tree_sitter.binding import Tree as TSTree

from cc_extractor.transform.base.base_context import BaseCodeContext

logger = logging.getLogger(__name__)
PY_MAX_LINE_LENGTH = 1000  # accepted max line length


class PythonCodeContext(BaseCodeContext):
    """
    Python code context
    """

    def __init__(self,
                 tree: TSTree,
                 code: str):
        super().__init__(tree, code, "python")
        if self.max_line_length <= PY_MAX_LINE_LENGTH:
            if not self.syntax_error:
                self.parsed_lines = self._parse()
            else:
                logger.info(f"Detect syntax error in code: {code[:50]} ..")

    def assign_token_to_line(self,
                             node: TSNode,
                             line_tokens: Dict[int, List]):
        if len(node.children) == 0 or node.type == "string":
            # leaf node or string node
            # NOTED: tree-sitter's Python string node is NOT a leaf node
            # for which it contains '(' and ')' as separate tokens, but not
            # the content of the string itself
            start_line_no = node.start_point[0]
            end_line_no = node.end_point[0]
            if start_line_no != end_line_no:
                logger.debug(f"Node {node} crossed different lines {start_line_no}, {end_line_no}")
                # in such case, it has to be a multiple line
                # string comment
                assert node.type == "string", \
                    f"Node type has to be string to cross line, get {node.type} instead"
                for l1 in range(start_line_no, end_line_no + 1):
                    line_tokens[l1].append(node)
            else:
                # the tokens should come in order
                line_tokens[start_line_no].append(node)
        else:
            for c in node.children:
                self.assign_token_to_line(c, line_tokens)

    def is_code_line(self, line_no) -> bool:
        code_line = self.parsed_lines[line_no]
        tokens = code_line.tokens
        if len(tokens) == 0:
            # empty line
            return False
        if len(tokens) == 1:
            if tokens[0].node.type == "string" or tokens[0].node.type == "comment":
                return False
        return True
