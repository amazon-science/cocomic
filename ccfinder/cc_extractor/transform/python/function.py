from asyncio.log import logger
from typing import List, Union, Optional

from tree_sitter import Node as TSNode
#from tree_sitter.binding import Node as TSNode

from cc_extractor.transform.base.base_function import BaseFunction
from cc_extractor.transform.python.context import PythonCodeContext


class PythonFunction(BaseFunction):

    def __init__(self,
                 node: TSNode,
                 context: PythonCodeContext):
        super().__init__(node, context)
        assert self.node.type == "function_definition"

    @property
    def colon_line_no(self):
        colon_node = None
        for c in self.node.children:
            if c.type == ":":
                colon_node = c
                break
        assert colon_node is not None
        return colon_node.start_point[0]

    @property
    def block(self) -> TSNode:
        """
        Usually block is the last element of a function
        """
        if self.node.children[-1].type != "block":
            raise RuntimeError(f"Node {self.node}'s last element isn't a block, children:{self.node.children}")
        return self.node.children[-1]

    @property
    def header_comments(self) -> List[TSNode]:
        # header comment is the one start at the
        # the beginning of a function, which serve as get_docstring, but
        # is not a get_docstring from parser's point of view. Just like
        # the one you are reading.
        #
        # for example
        # def f1(x):
        #     # this is the function's header comment
        children = self.node.children
        # locate : token
        colon_idx = None
        for i, c in enumerate(children):
            if c.type == ":":
                colon_idx = i
                break
        if colon_idx is None:
            raise RuntimeError(f"Function doesn't have ':', node: {self.node}")
        comments = list()
        for c_node in children[colon_idx + 1: -1]:
            # between : and block
            if c_node.type == "comment":
                comments.append(c_node)

        return comments

    def get_docstring(self, **kwargs) -> Union[TSNode, None]:
        """
        Docstring per definition should be the first string node
        that inside a block
        """

        def dfs_find_docstring(node):
            if node.type == "string":
                return node
            if len(node.children) == 0:
                # reach the end
                return None
            # recursively check whether the 1st node
            # is the string node
            return dfs_find_docstring(node.children[0])

        docstring_node = dfs_find_docstring(self.block)
        if docstring_node is not None:
            _doc_str = self.context.get_node_text(docstring_node)
            if _doc_str.startswith("\"\"\"") and _doc_str.endswith("\"\"\"") or \
                    _doc_str.startswith("'''") and _doc_str.endswith("'''"):
                # validate get_docstring to start with either """ or '''
                return docstring_node
        return None

    @staticmethod
    def get_documentation_end(comments: List[TSNode],
                              docstring_node: TSNode) -> int:
        if len(comments) == 0 and docstring_node is None:
            return -1
        doc_nodes = comments[:]
        if docstring_node is not None:
            doc_nodes.append(docstring_node)
        doc_end = max([dn.end_point[0] for dn in doc_nodes])
        return doc_end

    @property
    def body(self) -> List[str]:
        """
        Body excluding both self.header_comments and self.get_docstring
        """
        # function_start_line_no = self.node.start_point[0]
        function_end_line_no = self.node.end_point[0]

        comments_node = self.header_comments
        docstring_node = self.get_docstring()

        # print(comments_node)
        # print(docstring_node)

        if len(comments_node) == 0 and docstring_node is None:
            # in which case block is the body
            block = self.block
            code_lines = self.context.get_node_window(block)
            # print(code_lines)
            return [cl.text for cl in code_lines]

        # find the max end_line_no
        # among all doc nodes
        doc_end = self.get_documentation_end(comments_node, docstring_node)
        assert doc_end != -1

        body_start_line_no = doc_end + 1
        body_code_lines = self.context.get_window_by_line(body_start_line_no,
                                                          function_end_line_no + 1)
        return [cl.text for cl in body_code_lines]

    @property
    def name(self) -> Optional[str]:
        """
        Get function name
        """
        try:
            def_node = self.node.children[0]  # first node should be def or async
            assert def_node.type == "def" or def_node.type == "async", f"First child of a function node should be 'def' or 'async'," \
                f" get type {def_node.type} instead"
            # could be async
            if def_node.type == "def":
                func_name_idx = 1
            else:
                func_name_idx = 2
            func_name_node = self.node.children[func_name_idx]
            assert func_name_node.type == "identifier"
            return self.context.get_node_text(func_name_node)
        except Exception as e:
            logger.error('Error extracting function name: '+ str(e))
            return ""
    
    @property
    def decorator(self) -> str:
        """
        Get decorator of the function
        """
        prev_sibling_node = self.node.prev_sibling
        if prev_sibling_node is None:
            return ""
        if prev_sibling_node.type == "decorator":
            return self.context.get_node_text(prev_sibling_node)
        return ""
    
    @property
    def parameters(self) -> List[str]:
        """
        Get function input args
        possible param format: 
        basic: (param)
        default_parameter: (param = value)
        typed_parameter: (param : type)
        typed_default_parameter: (param : type = value)
        """
        
        try:
            def_node = self.node.children[0]  # first node should be def
            assert def_node.type == "def" or def_node.type == "async", f"First child of a function node should be 'def' or 'async'," \
                    f" get type {def_node.type} instead"
            
            if def_node.type == "def":
                func_param_idx = 2
            else:
                func_param_idx = 3
            func_param_node = self.node.children[func_param_idx]
            assert func_param_node.type == "parameters"
        except Exception as e:
            logger.error('Error getting function parameters: '+ str(e))
            return []

        # process the param lists. Text parsing is easier than tree-based parsing for parameters
        raw_params = self.context.get_node_text(func_param_node).strip("(").strip(")").replace("\n", "").replace(" ", "")
        raw_params = raw_params.split(",")
        params = []
        for r in raw_params:
            if r.strip() == "":
                continue
            p = {"name": "", "type": "", "default_value": ""}
            if ":" in r:
                p["name"], rest = r.split(":")[0], r.split(":")[1]
                if "=" in rest:
                    p["type"], p["default_value"] = rest.split("=")[0], rest.split("=")[1]
                else:
                    p["type"] = rest
            elif "=" in r:
                p["name"], p["default_value"] = r.split("=")[0], r.split("=")[1]
            else:
                p["name"] = r
            params.append(p)
        # except:
            # pass
        return params

    @property
    def return_type(self) -> Optional[str]:
        """
        Get function input args
        """
        try:
            def_node = self.node.children[0]  # first node should be def
            assert def_node.type == "def" or def_node.type == "async", f"First child of a function node should be 'def' or 'async'," \
                    f" get type {def_node.type} instead"
            
            if def_node.type == "def":
                func_return_type_idx = 4
            else:
                func_return_type_idx = 5
            func_return_type_node = self.node.children[func_return_type_idx]
            assert self.node.children[func_return_type_idx-1].type == "->" and func_return_type_node.type == "type"
            return self.context.get_node_text(func_return_type_node)
        except:
            return ""

    @property
    def return_statements(self):
            return_stmt_nodes = list()
            def _cb(n):
                return_stmt_nodes.append(n)
            self.context._dfs(self.node, ["return_statement"], _cb)
            return_stmt_texts = list()
            for node in return_stmt_nodes:
                return_stmt_texts.append(self.context.get_node_text(node))
            return list(set(return_stmt_texts))

    @property
    def assignments(self):
        """
        Get assignments within the function
        """
        assignment_stmt_nodes = list()
        def assignment_cb(n):
            assignment_stmt_nodes.append(n)
        self.context._dfs(self.node, ["assignment"], assignment_cb)
        assignments = [(self.context.get_node_text(node.children[0]), self.context.get_node_text(node.children[2])) for node in assignment_stmt_nodes]
        
        return assignments

    @staticmethod
    def find_all(context: PythonCodeContext) -> List['PythonFunction']:
        root_node = context.tree.root_node

        functions = list()

        def dfs(node: TSNode):
            if node.type == "function_definition":
                functions.append(PythonFunction(node, context))

            for child in node.children:
                dfs(child)

        dfs(root_node)

        return functions

    
