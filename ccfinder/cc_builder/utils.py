import datetime
import logging
import os
from import_deps import ModuleSet, ast_imports

EDGE_TYPE = [
    "project_file",
    "file_docstring",
    "global_var",
    "function",
    "func_parameter",
    "func_signature",
    "func_return_type",
    "func_decorator",
    "func_docstring",
    "func_body",
    "class",
    "class_baseclass_list",
    "class_signature",
    "class_docstring",
    "class_variable",
    "instance_variable",
    "member_function",
    "value",
    "default_value",
    "type",
    "use",
    "import",
    "project_file_reverse",
    "file_docstring_reverse",
    "global_var_reverse",
    "function_reverse",
    "func_parameter_reverse",
    "func_signature_reverse",
    "func_return_type_reverse",
    "func_decorator_reverse",
    "func_docstring_reverse",
    "func_body_reverse",
    "class_reverse",
    "class_baseclass_list_reverse",
    "class_signature_reverse",
    "class_docstring_reverse",
    "class_variable_reverse",
    "instance_variable_reverse",
    "member_function_reverse",
    "value_reverse",
    "default_value_reverse",
    "type_reverse",
    "use_reverse",
    "import_reverse"
]

PRUNED_EDGE_TYPE = [
    "project_file",
    "import",
    "file_docstring",
    "function",
    "class",
    "func_signature",
    "func_docstring",
    "class_signature",
    "class_docstring",
    "class_variable",
    "instance_variable"
    "member_function",
    "project_file_reverse",
    "import_reverse",
    "file_docstring_reverse",
    "function_reverse",
    "class_reverse",
    "func_signature_reverse",
    "func_docstring_reverse",
    "class_signature_reverse",
    "class_docstring_reverse",
    "class_variable_reverse",
    "instance_variable_reverse",
    "member_function_reverse",
]

REVERSE_EDGE_TYPE = [
    "function",
    "class",
    "class_variable",
    "instance_variable"
    "member_function",
    "global_var"
]

MODEL_NAME_MAPPING = {
    'codebert': "microsoft/codebert-base-mlm",
    'graphcodebert': "microsoft/graphcodebert-base",
    'unixcoder': "microsoft/unixcoder-base"
}

KEYWORDS_TO_IGNORE_NODES = ["Copyright", "Licence", "Author", "Version", "copyright", "licence", "author", "version", "COPYRIGHT", "LICENCE", "AUTHOR", "VERSION",]

def config_logging(level=None, prefix="log", log_dir_path=None):
    if not level:
        level = logging.DEBUG
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    # Avoid registering multiple Stream handlers to root logger, when many models exists under the single process.
    if root_logger.hasHandlers():
        return
    root_logger.setLevel(level)
    # config stdout
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(log_formatter)
    root_logger.addHandler(stdout_handler)
    # file handler
    if log_dir_path:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        logger_file = "%s_%s.log" % (prefix, timestamp)
        logger_file_path = os.path.join(log_dir_path, logger_file)
        file_handler = logging.FileHandler(logger_file_path)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


class ModuleSetForGraphRetrieval(ModuleSet):
    def __init__(self, path_list):
        super().__init__(path_list)
    def get_imports(self, module, return_fqn=False):
        imports = set()
        imports_list = list()
        raw_imports = ast_imports(module.path)
        for import_entry in raw_imports:
            # join 'from' and 'import' part of import statement
            full = ".".join(s for s in import_entry[:2] if s)

            import_level = import_entry[3]
            if import_level:
                # intra package imports
                intra = '.'.join(module.fqn[:-import_level] + [full])
                imported = self._get_imported_module(intra)
            else:
                imported = self._get_imported_module(full)

            if imported:
                imp = (imported.path, '.'.join(imported.fqn), import_entry)
                if imp not in imports:
                    imports.add(imp)
                    imports_list.append(imp)
        return imports_list

