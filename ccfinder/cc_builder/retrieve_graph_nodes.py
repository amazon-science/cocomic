import argparse
import logging
import ujson
import pathlib
import os
import pickle
import numpy as np
from multiprocessing import cpu_count, Pool
from collections import OrderedDict

from cc_builder.utils import config_logging, ModuleSetForGraphRetrieval, EDGE_TYPE


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
config_logging(logging.INFO)


def retrieve_khop_nodes(pinpoint_nodes, node_file, adj_file, k_hop=2):
    retrieved_nodes = list()
    retrieved_edges = dict()
    retrieved_edge_ids = list()
    node2id = {}
    with open(node_file, 'r') as f:
        nodes = f.readlines()
        for idx, n in enumerate(nodes):
            node2id[ujson.loads(n)] = idx
    with open(adj_file, "rb") as f:
        adj_info = pickle.load(f)
    
    id2node = adj_info["nodes"]
    node2id = {n: i for i, n in enumerate(id2node)}
    adj_mtx = adj_info["adj"]
    
    def get_khop_nodes_dfs(ppt_id, visited, k):
        # This is a graph DFS with depth limitation
        neighbors = []
        edges = []
        if k > 0:
            neighbors_1hop = list()
            edges_1hop = list()
            visited.add(ppt_id)
            nbs = np.nonzero(adj_mtx[:, ppt_id, :])
            neighbors_1hop += nbs[-1].tolist() # The non-zero actually keeps the order following the edge type
            edges_1hop += [{"head": ppt_id, "rel_id": nbs[0][i], "tail": nbs[1][i]} for i in range(len(nbs[0]))]
            # neighbors_1hop += np.nonzero(adj_mtx[:, :, ppt_id])[-1].tolist()
            # neighbors_1hop = sorted(list(set(neighbors_1hop)))
            for i, id in enumerate(neighbors_1hop):
                if id not in visited:
                    neighbors.append(id)
                    edges.append(edges_1hop[i])
                    n, e = get_khop_nodes_dfs(id, visited, k-1)
                    neighbors += n
                    edges += e
                    visited.add(id)
                    
        return neighbors, edges

    for ppt in pinpoint_nodes:
        if node2id.get(ppt) is None:
            logger.error(f"Cannot find node {ppt} in adjacency matrix")
            continue
        ppt_id = node2id[ppt]
        retrieved_ids, retrieved_eids = get_khop_nodes_dfs(ppt_id, set(), k_hop)
        assert len(set(retrieved_ids)) == len(retrieved_ids)
        retrieved_nodes += [ppt]
        retrieved_nodes += [id2node[i] for i in retrieved_ids]
        retrieved_edge_ids += retrieved_eids
    retrieved_nodes = list(OrderedDict.fromkeys(retrieved_nodes))
    
    retrieved_nodes_to_ids = {n: i for i, n in enumerate(retrieved_nodes)}

    # build edges
    edge_dict = dict()
    for e in retrieved_edge_ids:
        head = str(retrieved_nodes_to_ids[id2node[e["head"]]])
        if edge_dict.get(head) is None:
            edge_dict[head] = []
        edge_dict[head].append(((EDGE_TYPE[e["rel_id"]], str(e["rel_id"])), str(retrieved_nodes_to_ids[id2node[e["tail"]]])))

    for idx in range(len(retrieved_nodes)):
        if edge_dict.get(str(idx)) is not None:
            retrieved_edges[str(idx)] = edge_dict[str(idx)]
    final_retrieved_nodes = []
    for r in retrieved_nodes:
        final_retrieved_nodes.append(r)

    return final_retrieved_nodes, retrieved_edges


def retrieve_for_proj(proj, output_folder=None):
    proj_cxt_json, graph_folder = proj
    proj_retrieved_nodes = {}

    for f in os.listdir(graph_folder):
        if f.endswith(".node"):
            node_file = os.path.join(graph_folder, f)
        if f.endswith(".graph.adj.pk"):
            adj_file = os.path.join(graph_folder, f)
    assert os.path.exists(node_file) and os.path.exists(adj_file)

    with open(proj_cxt_json, 'r') as f:
        proj_dict = ujson.loads(f.read())
    if proj_dict["local_dep_stats"]["num_files_with_local_dep"] == 0:
        logger.info(f"This project, {proj_cxt_json}, does not have files with local dependencies")
        return
    proj_loc = proj_dict["project_location"] # This is the absolute path for project location

    proj_retrieved_nodes["project_location"] = proj_loc

    pkg_paths = pathlib.Path(proj_loc).glob('**/*.py')
    pkg_paths = [str(p) for p in pkg_paths]
    module_set = ModuleSetForGraphRetrieval(pkg_paths)

    proj_retrieved_nodes["retrived_nodes"] = {}
    proj_retrieved_nodes["retrieved_edges"] = {}
    for k in proj_dict["files_dependencies"].keys():
        if len(proj_dict["files_dependencies"][k]) != 0:
            print(f"Start processing {k}.")
            cur_file = os.path.join(proj_loc, k)
            local_imports = module_set.get_imports(module_set.by_path[cur_file])
            ppt_nodes = list()
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
                pinpoint_node = "".join(pinpoint_node.rsplit(".py", 1)) # remove .py at the end
                if entity is not None and entity != "*":
                    pinpoint_node += f".{entity}"
                ppt_nodes.append(pinpoint_node)        
            nodes, edges = retrieve_khop_nodes(ppt_nodes, node_file, adj_file)
            filtered_nodes = []
            for n in nodes:
                if n.startswith(k.split("/")[0]) or os.path.exists(os.path.join(proj_loc, n.split('.')[0])): # filter out index nodes
                    if ".self." in n:
                        n = n.replace(".self", "")
                    if "__init__." in n:
                        n = n.replace("__init__.", "")
                    filtered_nodes.append(".".join(n.split(".")[-2:]))
                else:
                    filtered_nodes.append(n)
            proj_retrieved_nodes["retrived_nodes"][k] = filtered_nodes
            proj_retrieved_nodes["retrieved_edges"][k] = edges

    if output_folder is None:
        output_folder = args.output_dir
    with open(os.path.join(output_folder, os.path.basename(proj_cxt_json).replace("project_context.json", f"retrieved_nodes.json")), 'w') as f:
        f.write(ujson.dumps(proj_retrieved_nodes, indent=2))

def main():
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if args.task == "single_proj":
        assert args.proj_cxt_json is not None
        retrieve_for_proj((args.proj_cxt_json, args.graph_folder))
    elif args.task == "batch_proj":
        assert args.proj_cxt_folder is not None
        proj_js_list = []
        graph_dir_list = []
        for g in os.listdir(args.graph_folder): # some projects do not have graph since they are timed out, so we use graph as index
            graph_folder = os.path.join(args.graph_folder, g)
            graph_dir_list.append(graph_folder)
            proj_js = os.path.join(args.proj_cxt_folder, g + "_project_context.json")
            proj_js_list.append(proj_js)
        assert len(proj_js_list) == len(graph_dir_list)
        with Pool(args.nprocs) as p:
            p.map(retrieve_for_proj, list(zip(proj_js_list, graph_dir_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=["single_proj", "batch_proj"])
    parser.add_argument('--proj_cxt_json', help='json file of (a single) project context')
    parser.add_argument('--proj_cxt_folder', help='folder of project context json files')
    parser.add_argument('--graph_folder', required=True, help="folder of graph structures")
    parser.add_argument('--output_dir', required=True, help='output directory to save project graphs')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

    args = parser.parse_args()
    main()