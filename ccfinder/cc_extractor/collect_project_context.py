# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
import logging

import ujson

import pathlib
from import_deps import ModuleSet, ast_imports

from cc_extractor.transform import load_parser, get_parse_context
from cc_extractor.transform.python.file_context.ast_file_context import PythonASTFileContext

from cc_extractor.utils import config_logging

import numpy as np
import os

from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
config_logging(logging.INFO)
    
ts_parser = load_parser("python", lib_path=os.path.join(os.getenv('PYTHONPATH'), "cc_extractor/build/python-lang-parser.so"))


class ProjectModuleSet(ModuleSet):
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
                if return_fqn:
                    imp = '.'.join(imported.fqn)
                else:
                    imp = imported.path
                if imp not in imports:
                    imports.add(imp)
                    imports_list.append(imp)
        return imports_list
        
    def get_non_local_imports(self, module):
        """return set of imported modules that are in self
        :param module: PyModule
        :return: (set - Path)
                 (set - str) if return_fqn == True
        """
        # print('####', module.fqn)
        # print(self.by_name.keys(), '\n\n')
        imports = set()
        raw_imports = ast_imports(module.path)
        for import_entry in raw_imports:
            # join 'from' and 'import' part of import statement
            full = ".".join(s for s in import_entry[:2] if s)

            import_level = import_entry[3]
            if import_level:
                intra = '.'.join(module.fqn[:-import_level] + [full])
                imported = self._get_imported_module(intra)
            else:
                imported = self._get_imported_module(full)

            if not imported:
                if import_entry[0] is not None:
                    imports.add(import_entry[0].split(".")[0] if "." in import_entry[0] else import_entry[0])
                else:
                    imports.add(import_entry[1].split(".")[0] if "." in import_entry[1] else import_entry[1])
                
        return imports


def analyze_file_dependency(project_dir):
    logger.info(f"Started file dependency extraction: {project_dir}")
    if not project_dir.endswith("/"):
        project_dir = project_dir + "/" 
    file_dependencies = {}
    non_local_imports = {}
    non_local_imports_set = set()
    pkg_paths = pathlib.Path(project_dir).glob('**/*.py')
    pkg_paths = [str(p) for p in pkg_paths]
    module_set = ProjectModuleSet(pkg_paths)

    local_dep_cnt = []
    
    for path in module_set.by_path.keys():
        try:
            local_imports = module_set.get_imports(module_set.by_path[path])
            local_imports = [str(i) for i in local_imports]
            local_imports = [i.replace(project_dir, "") for i in local_imports]
            file_dependencies[path.replace(project_dir, "")] = local_imports
            local_dep_cnt.append(len(local_imports))
        except Exception as e:
            logger.error('Error analyze local imports: '+ str(e))
        
        try:
            nlocal_imports = module_set.get_non_local_imports(module_set.by_path[path])
            for i in nlocal_imports:
                non_local_imports_set.add(i)
            non_local_imports[path.replace(project_dir, "")] = nlocal_imports
        except Exception as e:
            logger.error('Error analyze non local imports: '+ str(e))
    
    if len(local_dep_cnt) == 0:
        logger.warning("No local dependencies found")
        local_dep_stats = {"total_py_files": len(pkg_paths), "num_files_with_local_dep": np.count_nonzero(local_dep_cnt), "mean": 0, "std": 0, "max": 0}
    else:
        local_dep_stats = {"total_py_files": len(pkg_paths), "num_files_with_local_dep": np.count_nonzero(local_dep_cnt), "mean":  sum(local_dep_cnt)/len(local_dep_cnt), "std": np.std(np.array(local_dep_cnt)), "max": max(local_dep_cnt)}
    return {"project_location": project_dir, "files_dependencies": file_dependencies, "local_dep_stats": local_dep_stats, "non_local_imports": non_local_imports, "non_local_imports_set": non_local_imports_set, "might_miss_files": bool(non_local_imports_set & module_set.pkgs)}


def collect_file_context(file, project_prefix=None):
    logger.info(f"Started file-level context extraction for: {file}")
    file_context = {}
    if project_prefix is not None:
        file_context['file_path'] = file.replace(project_prefix, "")
    else:
        file_context['file_path'] = file
    with open(file) as f:
        code = f.read()
    parsed_context = get_parse_context(ts_parser, code, "python")

    if parsed_context.syntax_error:
        logger.info(f"Found syntax error in {file}, skip")
        return file_context

    ast_file_context = PythonASTFileContext(parsed_context) 
    file_context.update(ast_file_context.parse())
    
    return file_context

def collect_project_context(proj):
    input_project, output_file = proj
    try:
        project_context = analyze_file_dependency(input_project)
        project_context["project_context"] = []
        file_paths = pathlib.Path(input_project).glob('**/*.py')
        if not input_project.endswith("/"):
            project_prefix = input_project + "/"
        else:
            project_prefix = input_project
        for f in file_paths:
            f = str(f)
            project_context["project_context"].append(collect_file_context(f, project_prefix=project_prefix))
        with open(output_file, 'w') as f:
            f.write(ujson.dumps(project_context, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Error collecting project context: {e}")

def main():
    if args.task == "files_dependency":
        assert args.input_project is not None
        project_dependencies = analyze_file_dependency(args.input_project)
        with open(args.output, 'w') as f:
            f.write(ujson.dumps(project_dependencies, indent=2, ensure_ascii=False))
    elif args.task == "file_context":
        assert args.input_file is not None
        file_context = collect_file_context(args.input_file)
        with open(args.output, 'w') as f:
            f.write(ujson.dumps(file_context, indent=2, ensure_ascii=False))
    elif args.task == "project_context":
        assert args.input_project is not None
        collect_project_context((args.input_project, args.output))
    elif args.task == "batch_project_context":
        assert args.project_folder is not None
        proj_list = [os.path.join(args.project_folder, proj) for proj in os.listdir(args.project_folder)]
        output_list = [f"{args.output}/{proj}_project_context.json" for proj in os.listdir(args.project_folder)]
        assert len(proj_list) == len(output_list)
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        with Pool(args.nprocs) as p:
            p.map(collect_project_context, list(zip(proj_list, output_list)))
    else:
        raise NotImplementedError("task not implemented yet.")

        
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to collect project level context from python project")
    parser.add_argument("--task", required=True, choices=("files_dependency", "file_context", "project_context", "batch_project_context"), help="indicate the task for the tool")
    parser.add_argument("--project_folder", help="absolute path of the project folder for batch processing")
    parser.add_argument("--input_project", help="absolute path for the python project")
    parser.add_argument("--input_file", help="absolute path for the python file")
    parser.add_argument("--output", required=True, dest="output", help="task instances output path (folder or file)")
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    args = parser.parse_args()
    logger.info(f"Args: {args}")
    main()
