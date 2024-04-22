import argparse
import ujson
from collections import OrderedDict

EDGE_TYPES_TO_SQEEZE=["file_docstring", "value", "func_signature", "class_signature"]

def process_edges(retrieved_edges):
    proc_edges = []
    for head in retrieved_edges.keys():
        for e in retrieved_edges[head]:
            pe = [head, e[0][0], e[0][1], e[1]]
            proc_edges.append(pe)
    return proc_edges
    

def post_squeeze_nodes(nodes, edges):
    """
    Squeeze the retrieved nodes like ["file.func_name", "def func_name(): …"] 
    as one node ["#file.func_name\ndef func_name(): …"]
    """
    squeezed_nodes_map = {}
    squeezed_nodes = []
    squeezed_edges = []
    redundant_nodes = set()
    # TODO: This should be done in the retrieval step
    # de-duplicate edges
    edges = list(OrderedDict.fromkeys([tuple(e) for e in edges]))

    # create concrete edges with concrete nodes
    concrete_edges = [[nodes[int(e[0])], e[1], e[2], nodes[int(e[3])]] for e in edges]
    # merge the nodes and maintain the mapping
    for e in edges:
        head = nodes[int(e[0])]
        tail = nodes[int(e[3])]
        if e[1] in EDGE_TYPES_TO_SQEEZE:
            # add the index node to the leave nodes
            # ["file.func_name", "def func_name(): …"] -> ["#file.func_name\ndef func_name(): …"]
            new_node = "#" + head + "\n" + tail
            # Rare cases: nodes like "class.func_name" has more than one "def func_name(): …"
            # Example: adafruit-circuitpython-progressbar_-_YYYY-MM-DD_data/adafruit-circuitpython-progressbar-2.3.0/adafruit_progressbar/progressbar.py
            if squeezed_nodes_map.get(head) is None:
                squeezed_nodes_map[head] = []
            if squeezed_nodes_map.get(tail) is None:
                squeezed_nodes_map[tail] = []
            squeezed_nodes_map[head].append(new_node)
            squeezed_nodes_map[tail].append(new_node)
            redundant_nodes.add(tail)
    # update the edges with the new nodes
    for e in concrete_edges:
        head = e[0]
        tail = e[3]
        heads, tails = None, None
        if head in squeezed_nodes_map.keys():
            heads = squeezed_nodes_map[head]
        if tail in squeezed_nodes_map.keys():
            tails = squeezed_nodes_map[tail]
        if heads is None:
            heads = [head]
        if tails is None:
            tails = [tail]
        for h in heads:
            for t in tails:
                if t not in heads: # ignore self loop:
                    squeezed_edges.append([h, e[1], e[2], t])
    # update the nodes with the new nodes
    for n in nodes:
        if n not in redundant_nodes:
            if n in squeezed_nodes_map.keys():
                squeezed_nodes += squeezed_nodes_map[n]
            else:
                squeezed_nodes.append(n)
    squeezed_nodes2idx = {n: i for i, n in enumerate(squeezed_nodes)}
    # convert nodes in edges to indices
    squeezed_edges = [[str(squeezed_nodes2idx[e[0]]), e[1], e[2], str(squeezed_nodes2idx[e[3]])] for e in squeezed_edges]
    squeezed_nodes = [s.strip() for s in squeezed_nodes]

    return squeezed_nodes, squeezed_edges

def create_test_samples(pair):
    samples = []
    
    prompt_file, retrieved_node_file = pair
    # print(prompt_file)
    with open(retrieved_node_file, 'r') as f:
        retrieved_info = ujson.load(f)
    proj_loc = retrieved_info["project_location"]
    retrieved_nodes = retrieved_info["retrived_nodes"]
    retrieved_edges = retrieved_info["retrieved_edges"]

    with open(prompt_file, 'r') as f:
        lines = f.readlines()
        prompts = [ujson.loads(l) for l in lines]

    for p in prompts:
        s = {}
        s["prompt"] = p["prompt"]
        s["groundtruth"] = p["groundtruth"]
        s["retrieved_nodes"] = retrieved_nodes[p["metadata"]["file"].replace(proj_loc, "")]
        if len(s["retrieved_nodes"]) == 0:
            continue
        s["retrieved_edges"] = process_edges(retrieved_edges[p["metadata"]["file"].replace(proj_loc, "")])
        # post-processing for futher compressing nodes
        s["retrieved_nodes"], s["retrieved_edges"] = post_squeeze_nodes(s["retrieved_nodes"], s["retrieved_edges"])
        if len(s["retrieved_edges"]) == 0:
            continue
        s["metadata"] = p["metadata"]
        samples.append(s)

    return "\n".join([ujson.dumps(samp) for samp in samples])

def prepend_locale(edges):
    """
    Collect entities and prepend the locale to each entity.
    This method is only used for random entity experiments.
    """
    entities = []

    for e in edges:
        if e["rel"] in EDGE_TYPES_TO_SQEEZE:
            entities.append(e["head"])
            entity = "#" + e["head"] + "\n" + e["tail"]
            entities.append(entity)
        else:
            entities.append(e["head"])
            entities.append(e["tail"])
    return list(OrderedDict.fromkeys(entities))


def main():
    with open(args.output_file, 'w') as f:
        f.write(create_test_samples((args.prompt_file, args.retrieved_entity_file)) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to create data split")
    parser.add_argument("--retrieved_entity_file", required=True, help="the folder of project context with retrieved nodes (absolute path)")
    parser.add_argument("--prompt_file", required=True, help="the folder of project context json files (absolute path)")
    parser.add_argument("--output_file", required=True, help="the output json file of model input")
    args = parser.parse_args()
    main()