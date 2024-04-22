import networkx as nx
from multiprocessing import cpu_count, Pool
import argparse
import ujson
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import pickle
import os
import logging
import torch
import shutil

from cc_builder.utils import EDGE_TYPE, PRUNED_EDGE_TYPE, config_logging, MODEL_NAME_MAPPING, REVERSE_EDGE_TYPE, KEYWORDS_TO_IGNORE_NODES
from cc_builder.utils import ModuleSetForGraphRetrieval
import pathlib


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
config_logging(logging.INFO)


def construct_adj_matrix(input_graph_path, input_nodes_path, output_adj_path, encode=False, node_emb_lm=None, batch_size=None, max_seq_length=None):
    logger.info(f'Create adjacency matrix for {os.path.basename(input_nodes_path).split(".")[0]}...')
    proj_graph = nx.read_gpickle(input_graph_path)
    with open(input_nodes_path, 'r') as f:
        nodes = [ujson.loads(l) for l in f.readlines()]
    n_rel = len(EDGE_TYPE)
    n_node = len(nodes)
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)

    for s in range(n_node):
        for t in range(n_node):
            if proj_graph.has_edge(s, t):
                for e_attr in proj_graph[s][t].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    
    assert proj_graph.number_of_edges() == np.count_nonzero(adj)
    
    adj_mtx = {'adj': adj, 'nodes': nodes}
    logger.info(f'Adjacency matrix saved to {output_adj_path}.')
    with open(output_adj_path, 'wb') as fout:
        pickle.dump(adj_mtx, fout)


def construct_graph(edge_jsonl_path, node_path, output_graph_path):
    """
    Construct graph and save it into pickle file
    """
    logger.info(f'Construct project context graph for {os.path.basename(node_path).split(".")[0]}...')
    node2id = {}
    id2node = {}
    with open(node_path, "r", encoding="utf8") as fin:
        id2node = [ujson.loads(w) for w in fin]
    node2id = {w: i for i, w in enumerate(id2node)}
    edge2id = {r: i for i, r in enumerate(EDGE_TYPE)}

    graph = nx.MultiDiGraph()
    
    with open(edge_jsonl_path, "r", encoding="utf8") as fin:
        attrs = set()
        lines = fin.readlines()
        for l in lines:
            line = ujson.loads(l)
            rel = edge2id[line["rel"]]
            head = node2id[line["head"]]
            tail = node2id[line["tail"]]
            if (head, rel, tail) not in attrs:
                graph.add_edge(head, tail, rel=rel)
                attrs.add((head, rel, tail))
                # reverse edge
                if EDGE_TYPE[rel] in REVERSE_EDGE_TYPE:
                    graph.add_edge(tail, head, rel=rel + int(len(edge2id) / 2))
                    attrs.add((tail, rel + int(len(edge2id) / 2), head))
    
    assert len(attrs) == graph.number_of_edges()

    nx.write_gpickle(graph, output_graph_path)
    logger.info(f"graph file saved to {output_graph_path}")
       

def extract_node_edge_coarse_grained(proj_cxt, output_edge_path, output_nodes_path):
    """
    Top-down extract nodes and edges from project context dict.
    """
    proj_loc = proj_cxt["project_location"]
    logger.info(f'Extract node and edges for {proj_loc}...')

    nodes = []
    node_seen = set()
    node_cnt = {"project": 0, 
                "file_name": 0, 
                "global_var": 0,
                "global_var_value": 0,
                "file_docstring": 0,
                "func_name": 0, 
                "func_signature": 0, 
                "func_docstring": 0,
                "func_param_name": 0,
                "func_param_type": 0,
                "func_param_value": 0,
                "func_return_type": 0,
                "func_decorator": 0,
                "func_body": 0,
                "class_name": 0,
                "class_signature": 0,
                "class_baseclass": 0,
                "class_docstring": 0,
                "class_var": 0,
                "class_var_name": 0,
                "class_var_type": 0,
                "class_var_value": 0,
                "instance_var": 0,
                "unk": 0
                }

    pkg_paths = pathlib.Path(proj_loc).glob('**/*.py')
    pkg_paths = [str(p) for p in pkg_paths]
    module_set = ModuleSetForGraphRetrieval(pkg_paths)

    with open(output_edge_path, 'w', encoding="utf8") as fout:

        def add_node_edge(check_node, node_type, head, rel, tail):
            """
            check_node is the node to be added to the seen, and it should be part of the new edge
            """
            added_node = False 
            added_edge = False
            # Above is just for debugging
            assert check_node == head or check_node == tail
            assert rel in EDGE_TYPE
            if check_node not in node_seen and check_node != "":
                nodes.append(check_node)
                node_seen.add(check_node)
                node_cnt[node_type] += 1
                added_node = True
            if head != "" and tail != "":
                assert head in node_seen and tail in node_seen
                fout.write(ujson.dumps({"head": head, "rel": rel, "tail": tail}, ensure_ascii=False) + '\n')
                added_edge = True
            return added_node, added_edge

        def build_function(func_list, func_parent, rel="function"):
            """
            extract node and edges for a list of functions
            func_list: the list contains dict format function details
            func_parent: the parent node of function nodes
            """
            for func in func_list:
                func_node = func_parent + "." + func["func_name"]
                # edge: file -> function
                add_node_edge(func_node, "func_name", func_parent, rel, func_node)
                # node: func_details
                func_details = func["func_orig_str"]
                add_node_edge(func_details, "func_signature", func_node, "func_signature", func_details)
                
        for fk in proj_cxt["files_dependencies"].keys():
            file_node_k = fk.replace("/", ".")
            file_node_k = "".join(file_node_k.rsplit(".py", 1)) # remove .py at the end
            # create node without edges
            add_node_edge(file_node_k, "file_name", "", "project_file", file_node_k)

        for fc in proj_cxt["project_context"]:
            # Syntax error or timeout happens when we parse some files, and they do not build file context for them. We also skip them when building the graph
            if len(fc) == 1:
                continue
            file_node = fc["file_path"].replace("/", ".")
            file_node = "".join(file_node.rsplit(".py", 1))
            try:
                assert file_node in node_seen, f"Didn't find {file_node} when building file nodes." # this is a validation, but might not have real effects on the final graph
            except:
                nodes.append(file_node)
                node_seen.add(file_node)
            # node: file_docstring
            file_docstr_node = fc["file_docstring"]
            # ignore useless docstring
            if not any(k in file_docstr_node for k in KEYWORDS_TO_IGNORE_NODES):
                # edge: file -> file_docstring
                add_node_edge(file_docstr_node, "file_docstring", file_node, "file_docstring", file_docstr_node)
            # node: global variables
            # we ignore global variable for now. will add back if necessary
            # """
            for gv in fc["global_vars"]:
                glob_var_node = file_node + "." + gv[0]
                glob_var_value = gv[0] + "=" + gv[1]
                # ignore useless global variables
                if not any(k in glob_var_value for k in KEYWORDS_TO_IGNORE_NODES):
                    # file -> global_var
                    add_node_edge(glob_var_node, "global_var", file_node, "global_var", glob_var_node)
                    # global_var -> value
                    add_node_edge(glob_var_value, "global_var_value", glob_var_node, "value", glob_var_value)
            # """
            # node: functions
            build_function(fc["functions"], file_node)
            
            for cls in fc["classes"]:
                cls_node = file_node + "." + cls["class_name"]
                # edge: file -> class
                add_node_edge(cls_node, "class_name", file_node, "class", cls_node)
                # node: class_signature
                cls_detail_node = cls["class_signature"] + "\n" + cls["class_docstring"]
                cls_var_node = ", \n".join(cls["class_variables"]["raw"])
                cls_detail_node += "\n" + cls_var_node
                
                # edge: class -> member_function
                add_node_edge(cls_detail_node, "class_signature", cls_node, "class_signature", cls_detail_node)
                build_function(cls["functions"], cls_node, "member_function")
        
        for k in proj_cxt["files_dependencies"].keys():
            file_node_k = k.replace("/", ".")
            file_node_k = "".join(file_node_k.rsplit(".py", 1))
            if len(proj_cxt["files_dependencies"][k]) != 0:
                cur_file = os.path.join(proj_loc, k)
                local_imports = module_set.get_imports(module_set.by_path[cur_file])
                for imp in local_imports:
                    module_path, module_name, raw_import = imp
                    entity = None
                    if raw_import[0] is not None:
                        # this is from xxx import yy, yy can be a file or stuff in the file
                        if module_name.endswith(raw_import[1]):
                            pkg = raw_import[1] # yy is a file
                        else:
                            pkg = raw_import[0]
                            entity = raw_import[1]
                    else:
                        pkg = raw_import[1]
                    try:
                        assert pkg in module_name or module_name.endswith("__init__")
                    except:
                        logger.warning(f"module name does not match raw import: {module_name}, {raw_import}. Skipping.")
                        continue
                    pinpoint_node = str(module_path).replace(proj_loc, "").replace("/", ".")
                    pinpoint_node = "".join(pinpoint_node.rsplit(".py", 1))
                    if entity is not None and entity != "*":
                        pinpoint_node += f".{entity}"
                    add_node_edge(pinpoint_node, "unk", file_node_k, "import", pinpoint_node) # this is the imported stuff in each file, we do not know the node type
                    # cover the case that fileC: "from fileA import classB", fileD: "from fileC import classB". Class B is not defined in fileA but imported.
                    # new node: fileA.classB
                    if entity == "*":
                        # * means all the stuff in the file, so we link imported node with all the 1st hierarchy stuff in the file
                        for n in nodes:
                            # have to traverse all the nodes again, might be time consuming, but * is a rare corner case we hope to cover
                            # startswith pinpoint_node means the node is in the same file
                            # split('.') == 2 means the node is in the 1st hierarchy
                            if n.startswith(pinpoint_node) and len(n.replace(pinpoint_node, "").split(".")) == 2:
                                imported_node = file_node_k + "." + n.split(".")[-1]
                                add_node_edge(imported_node, "unk", imported_node, "import", n)
                    else:
                        imported_node = file_node_k + "." + pinpoint_node.split(".")[-1]
                        add_node_edge(imported_node, "unk", imported_node, "import", pinpoint_node)
 
    assert len(node_seen) == len(nodes)
    with open(output_nodes_path, 'w') as nout:
        nout.write("\n".join([ujson.dumps(n, ensure_ascii=False) for n in nodes]))
    logger.info(f'Finished node and edge extraction for {proj_cxt["project_location"]}...')
        

def create_cxt_graph(proj):
    proj_cxt_json, output_dir = proj
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    node_file = os.path.join(output_dir, os.path.basename(proj_cxt_json).replace(".json", ".node"))
    edge_file = os.path.join(output_dir, os.path.basename(proj_cxt_json).replace(".json", ".edge.jsonl"))
    graph_file = os.path.join(output_dir, os.path.basename(proj_cxt_json).replace(".json", ".graph"))
    adj_file = os.path.join(output_dir, os.path.basename(proj_cxt_json).replace(".json", ".graph.adj.pk"))
    with open(proj_cxt_json, 'r') as f:
        proj_cxt = ujson.load(f)

    extract_node_edge_coarse_grained(proj_cxt, edge_file, node_file)

    # Filter out projects with too many nodes in the graph (memory/disk-consuming when building the adj matrix)
    # Un-comment the following line if you want to batchprocess a dataset and add length filter as the paper does
    # try:
        # with open(node_file, 'r') as f:
        #     num_node = len(f.read().split("\n"))
        # if args.max_node_num is not None:
        #     assert num_node < args.max_node_num
        # if args.min_node_num is not None:
        #     assert num_node > args.min_node_num
    # except:
    #     logger.warning(f"Too many nodes or too few nodes in {proj_cxt_json}, skipping.")
    #     shutil.rmtree(output_dir)
    #     return

    construct_graph(edge_file, node_file, graph_file)
    construct_adj_matrix(graph_file, node_file, adj_file)
        
def main():
    if args.task == "single_proj":
        assert args.proj_cxt_json is not None
        create_cxt_graph((args.proj_cxt_json, args.output_dir))
    elif args.task == "batch_proj":
        assert args.proj_cxt_folder is not None
        proj_js_list = []
        output_dir_list = []
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        for js in os.listdir(args.proj_cxt_folder):
            proj_js = os.path.join(args.proj_cxt_folder, js)
            proj_js_list.append(proj_js)
            output_dir = os.path.join(args.output_dir, js.replace("_project_context.json", ""))
            output_dir_list.append(output_dir)
        assert len(proj_js_list) == len(output_dir_list)
        with Pool(args.nprocs) as p:
            p.map(create_cxt_graph, list(zip(proj_js_list, output_dir_list)))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=("single_proj", "batch_proj"))
    parser.add_argument('--proj_cxt_json', help='json file of (a single) project context')
    parser.add_argument('--proj_cxt_folder', help='folder of project context json files')
    parser.add_argument('--output_dir', required=True, help='output directory to save project graphs')
    parser.add_argument('--node_emb_lm', default="codebert", help="pre-trained model to embed node content")
    parser.add_argument('--node_emb_bs', default=2, help="batch size for node encoding")
    parser.add_argument('--max_seq_per_node', default=32, help="max sequence length for node content")
    # max_node_num filters out projects that take "too much memory/disk" to build the adjacency matrix
    # adjacency matrix is a np.array of shape (max_node_num, max_node_num, 44), 44 is the number of edge types
    # the value type in adjacency matrix is np.uint8, which takes 1 byte, so the max memory/disk usage is
    # (44 * max_node_num * max_node_num) bytes
    # default value 5000 will result in a max memory/disk usage of 44 * 5000 * 5000 = ~1.1GB graph.adj.pk file
    parser.add_argument('--min_node_num', default=10, type=int, help='min number of nodes in the project')
    parser.add_argument('--max_node_num', default=5000, type=int, help='max number of nodes in the project')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

    args = parser.parse_args()

    main()
