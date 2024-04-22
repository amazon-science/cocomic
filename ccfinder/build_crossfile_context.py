# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import argparse
from cc_extractor.collect_project_context import collect_project_context
import os
from cc_builder.build_context_graph import create_cxt_graph
from cc_builder.retrieve_graph_nodes import retrieve_for_proj

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to collect project level context from python project")
    parser.add_argument("--input_project", help="absolute path for the python project")
    parser.add_argument("--output_dir", help="output folder")
    args = parser.parse_args()

    assert os.path.exists(args.input_project), f"Input project {args.input_project} does not exist"
    args.input_project = args.input_project.rstrip("/")
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    # output file for project information in json format
    proj_info_file = os.path.join(args.output_dir, os.path.basename(args.input_project)) + "_project_context.json"
    collect_project_context((args.input_project, proj_info_file))
    # create project-level context graph
    create_cxt_graph((proj_info_file, args.output_dir))
    # retrieve relevant entities
    retrieve_for_proj((proj_info_file, args.output_dir), args.output_dir)
