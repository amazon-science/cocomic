# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

import datetime
import logging
import os
import random
from typing import Dict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def str2bool(v):
    """
    Check whether v is a legit boolean argument
    """
    return v.lower() in ("yes", "true", "t", "1")


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


def get_jsonl_code(doc: Dict, lang):
    content_lines = doc.get('content', [])
    if len(content_lines) > 0:
        # code is usually the 1st item in contents
        code = content_lines[0]
        if code['type'] == "code" and code['lang'] == lang:
            return code
    return None


def get_code_metadata(jsonl_doc: Dict):
    url = jsonl_doc.get("url", None)
    doc_id = jsonl_doc.get("id", None)
    file_path = jsonl_doc.get("filepath", None)
    repo = jsonl_doc.get("repository", None)

    return {
        "url": url,
        "id": doc_id,
        "filepath": file_path,
        "repository": repo
    }


def reservoir_sampling(iterable, n):
    """
    Returns @param n random items from @param iterable.
    """
    reservoir = []
    for t, item in enumerate(iterable):
        if (t + 1) % 100000 == 0:
            logger.info(f"Iterate through {t + 1} lines")
        if t < n:
            reservoir.append((t, item))
        else:
            m = random.randint(0, t)
            if m < n:
                reservoir[m] = (t, item)
    return reservoir


def sample_file_lines(fn: str, size: int):
    logger.info(f"Sample {size} lines from {fn}")

    def line_iterator():
        with open(fn, encoding="utf8") as f:
            for line in f:
                yield line

    return reservoir_sampling(line_iterator(), size)


def split_code_lines(user_context_code: str):
    """
    Break code into lines
    """
    lines = user_context_code.rstrip().split("\n")
    return lines
