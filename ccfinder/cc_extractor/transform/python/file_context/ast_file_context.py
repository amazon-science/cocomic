from cc_extractor.transform.base.base_file_context import BaseFileContext
from cc_extractor.transform.python.context import PythonCodeContext

from cc_extractor.transform.python.function import PythonFunction
from cc_extractor.transform.python.pyclass import PythonClass

import warnings
class PythonASTFileContext(BaseFileContext):

    def __init__(self, context: PythonCodeContext):
        super().__init__(context)

    def file_docstrings(self, node) -> str:
        """
        Collect the docstring for files
        File docstring should be in the first hierarchy of the file
        and the first valid string node
        """
        for child in node.children:
            if child.type == "expression_statement":
                for c in child.children:
                    if c.type == "string":
                        _doc_str = self.context.get_node_text(c)
                        if _doc_str.endswith("\"\"\"") and _doc_str.endswith("\"\"\"") or \
                            _doc_str.endswith("r\"\"\"") and _doc_str.endswith("\"\"\"") or \
                            _doc_str.startswith("'''") and _doc_str.endswith("'''") or \
                            _doc_str.startswith("r'''") and _doc_str.endswith("'''"):
                            return _doc_str
        return ""
    
    def docstrings(self, node) -> str:
        """
        Collect the docstring for classes and functions

        Class/function Docstring should be the first string node
        that inside the file
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
        docstring_node = dfs_find_docstring(node)
        if docstring_node is not None:
            _doc_str = self.context.get_node_text(docstring_node)
            # capture """docstring""", '''docstring''', r'''docstring''', r"""docstring"""
            if _doc_str.endswith("\"\"\"") and _doc_str.endswith("\"\"\"") or \
                _doc_str.endswith("r\"\"\"") and _doc_str.endswith("\"\"\"") or \
                _doc_str.startswith("'''") and _doc_str.endswith("'''") or \
                _doc_str.startswith("r'''") and _doc_str.endswith("'''"):
                # validate get_docstring to start with either """ or '''
                return _doc_str
        return ""
    
    
    def imports(self):
        """
        Collect all the import statements from the file

        For example:
        1. import system
        2. import a from sys
        3. from system import utils as s
        """
        import_nodes = self.collect_nodes([
            "import_statement",
            "import_from_statement"
        ])
        # format each node into blocks
        # which include all the lines from the code block
        import_blocks = [self.get_code_block(n) for n in import_nodes]

        return {
            "imports": [b.text for b in import_blocks]
        }

    def global_variables(self):
        """
        Collect global variables in the file
        We assume that global variables should not have duplicated names
        """
        
        global_vars = list()
        # func_nodes = self.collect_nodes([
        #     "function_definition"
        # ])
        # if len(func_nodes) > 0:
        for child in self.context.tree.root_node.children:
            if child.type == "expression_statement":
                for c in child.children:
                    if c.type == "assignment" and len(c.children) == 3:
                        var = self.context.get_node_text(c.children[0])
                        value = self.context.get_node_text(c.children[2])
                        global_vars.append([var, value])
        return {"global_vars": global_vars}

    def functions(self, node, k="functions", additional_ids=None):
        """
        Collect all functions in the first hierarchy of the node
        """
        def find_functions(node):
            func_nodes = []
            for child in node.children:
                if child.type == "function_definition":
                    func_nodes.append(child)
                else: # check one further hierarchy to get the decorated functions
                    for c in child.children:
                        if c.type == "function_definition":
                            func_nodes.append(c)
            return func_nodes

        def extract_use(assignments, ids):
            uses = dict()
            for asn in assignments:
                left = asn[0]
                right = asn[1]
                left_exist = False
                for id in ids:
                    if id == left:
                        left_exist = True
                        k = id
                        break
                if not left_exist:
                    continue
                for id in ids:
                    if id in right and id != k and id != "self" and id != "*":
                        if uses.get(k) is None:
                            uses[k] = []
                        uses[k].append(id)
            return uses
        
        func_nodes = find_functions(node)
        func_blocks = list()
        for fn in func_nodes:
            py_func = PythonFunction(fn, self.context)
            func_name = py_func.name
            func_params = py_func.parameters
            func_return_type = py_func.return_type
            func_decorator = py_func.decorator
            docstring_node = py_func.get_docstring()
            func_docstring = self.context.get_node_text(docstring_node) if docstring_node is not None else ""
            func_body = py_func.body
            func_return_stmts = py_func.return_statements
            func_assignments = py_func.assignments

            ids = list(p["name"] for p in func_params)
            if additional_ids is not None:
                ids += additional_ids
            use_dependencies = extract_use(func_assignments, ids)
            # find the : line number
            colon_lineno = py_func.colon_line_no
            fn_st_lineno = fn.start_point[0]
            fn_signature_lines = self.context.get_window_by_line(fn_st_lineno, colon_lineno + 1)

            func_blocks.append(
                {
                    "func_signature": "\n".join([nl.text for nl in fn_signature_lines]),
                    "func_name": func_name,
                    "func_parameters": func_params,
                    "func_return_type": func_return_type,
                    "func_decorator": func_decorator,
                    "func_docstring": func_docstring,
                    "func_body": "\n".join(func_body),
                    "return_statements": func_return_stmts,
                    "func_orig_str": self.context.get_node_text(fn),
                    "byte_span": [fn.start_byte, fn.end_byte],
                    "start_point": fn.start_point,
                    "end_point": fn.end_point,
                    "use_dependencies": use_dependencies
                }
            )

        return {
            k: func_blocks
        }

    def classes(self):
        """
        Collect all classes.
        """
        class_nodes = self.collect_nodes([
            "class_definition"
        ])
        class_blocks = list()
        for cn in class_nodes:
            py_class = PythonClass(cn, self.context)
            class_name=py_class.name
            class_argument_list = py_class.arguments
            instance_variables = py_class.instance_variable
            class_variables = py_class.class_variable       
            try:
                class_body_node = cn.children[-1]
                assert class_body_node.type == "block"  # this might fail sometimes. Cover the corner cases once failed
            except:
                warnings.warn(f"The class {class_name} might not have body")
                continue
            class_docstring = self.docstrings(class_body_node)
            member_functions = self.functions(class_body_node, additional_ids=instance_variables)
            
            # find the first :
            st_lineno = cn.start_point[0]
            colon_node = None
            for c in cn.children:
                if c.type == ":":
                    colon_node = c
                    break
            assert colon_node is not None
            ed_lineno = colon_node.start_point[0]
            node_lines = self.context.get_window_by_line(st_lineno, ed_lineno + 1)
            class_block = {
                    "class_signature": "\n".join([nl.text for nl in node_lines]),
                    "class_name": class_name,
                    "class_baseclass_list": class_argument_list,
                    "class_docstring": class_docstring,
                    "class_variables": class_variables,
                    "instance_variables": instance_variables
                }
            class_block.update(member_functions)
            class_block.update({"class_orig_str": self.context.get_node_text(cn)})
            class_block.update({"byte_span": [cn.start_byte, cn.end_byte], "start_point": cn.start_point, "end_point": cn.end_point})
            class_blocks.append(class_block)
            

        return {
            "classes": class_blocks
        }

    def parse(self):
        """
        Return consolidated file context including, imports/function definitions/class definitions
        """
        file_context = dict()
        file_context.update({"file_docstring": self.file_docstrings(self.context.tree.root_node)})
        file_context.update(self.imports())
        file_context.update(self.global_variables())
        file_context.update(self.functions(self.context.tree.root_node))
        file_context.update(self.classes())
        return file_context