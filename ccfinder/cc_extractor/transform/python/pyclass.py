from typing import List, Optional

from tree_sitter.binding import Node as TSNode

from cc_extractor.transform.python.context import PythonCodeContext


class PythonClass:

    def __init__(self,
                 node: TSNode,
                 context: PythonCodeContext):
        self.node = node
        self.context = context
        assert self.node.type == "class_definition"

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
    def arguments(self) -> Optional[str]:
        """
        Get class argument list
        """
        args = list()
        try:
            class_node = self.node.children[0]  # first node should be def
            assert class_node.type == "class", f"First child of a function node should be 'class'," \
                f" get type {class_node.type} instead"
            # could be async
            class_argument_node = self.node.children[2]
            assert class_argument_node.type == "argument_list"
            # process the arg lists
            raw_args = self.context.get_node_text(class_argument_node).strip("(").strip(")").replace("\n", "").replace(" ", "")
            args = raw_args.split(",")
        except:
            pass
        return args


    @property
    def name(self) -> Optional[str]:
        """
        Get class name
        """
        try:
            class_node = self.node.children[0]  # first node should be class
            assert class_node.type == "class", f"First child of a class node should be 'class'," \
                f" get type {class_node.type} instead"
            # could be async
            class_name_node = self.node.children[1]
            assert class_name_node.type == "identifier"
            return self.context.get_node_text(class_name_node)
        except:
            return None

    @property
    def instance_variable(self) -> List[str]:
        """
        Get the instance variable of the class, i.e., self.var = ...
        """
        instance_variables = list()

        init_func_nodes = list()
        def _cb(n):
            if n.children[1].type == "identifier" and self.context.get_node_text(n.children[1]) == "__init__":
                init_func_nodes.append(n)
        self.context._dfs(self.node, ["function_definition"], _cb)
        try:
            assert len(init_func_nodes) == 1
        except:
            return instance_variables

        attribute_nodes = list()
        def _cb(n):
            attribute_nodes.append(n)
        self.context._dfs(init_func_nodes[0], ["attribute"], _cb)
        
        for node in attribute_nodes:
            try:
                assert self.context.get_node_text(node.children[0]) == "self"
            except:
                continue
            node_text = self.context.get_node_text(node)
            if node_text not in instance_variables:
                instance_variables.append(node_text)
        return instance_variables

    @property
    def class_variable(self) -> List[str]:
        """
        Get the class variable of the class, i.e., variable outside the constructor and share by all instances
        """
        class_variables = {"raw": [], "processed":[]}
        try:
            class_body_node = self.node.children[-1]
            assert class_body_node.type == "block"  # this might fail sometimes. Cover the corner cases once failed
        except:
            import warnings
            warnings.warn(f"The class might not have body")
            return class_variables
        for child in class_body_node.children:
            if child.type == "expression_statement":
                for c in child.children:
                    if c.type == "assignment":
                        class_variables["raw"].append(self.context.get_node_text(c))
                        if len(c.children) == 3:
                            class_variables["processed"].append({"name": self.context.get_node_text(c.children[0]), "type": "", "value": self.context.get_node_text(c.children[2])})
                        elif len(c.children) == 5:
                            class_variables["processed"].append({"name": self.context.get_node_text(c.children[0]), "type": self.context.get_node_text(c.children[2]), "value": self.context.get_node_text(c.children[4])})       
        return class_variables

    @staticmethod
    def find_all(context: PythonCodeContext) -> List['PythonClass']:
        root_node = context.tree.root_node

        classes = list()

        def dfs(node: TSNode):
            if node.type == "class_definition":
                classes.append(PythonClass(node, context))

            for child in node.children:
                dfs(child)

        dfs(root_node)

        return classes
