import json
import os
from typing import Tuple

import faiss
import scipy
from tqdm import tqdm
import numpy as np

def prepare_dataset(args : dict) -> Tuple[list, dict, dict, dict, dict, np.array, np.array, faiss.Index, dict, dict, list, list, list, dict, scipy.sparse.csr_matrix]:
    """prepare dataset

    Args:
        args (dict): args

    Returns:
        Tuple[list, dict, dict, dict, dict, np.array, np.array, faiss.Index, dict, dict, list, list, list, dict, scipy.sparse.csr_matrix]: itemid_modalities, passages_dict, tables_dict, images_dict, images_titles, item_ids, item_reps, gpu_index, qrels, item_id_to_idx, qrels_data, qrels_row_idx, qrels_col_idx, qid_to_idx, qrels_sparse_matrix
    """
    itemid_modalities = []
    
    passages_dict = prepare_passages(args.passages_file, itemid_modalities)
    
    tables_dict = prepare_tables(args.tables_file, itemid_modalities)

    images_dict = prepare_images(args.images_file, args.images_path, itemid_modalities)

    image_answer_set = set()
    with open(args.train_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            image_answer_set.add(json.loads(line.strip())['answer'][0]['answer'])
    
    image_answers_str = ''
    for s in image_answer_set:
        image_answers_str = image_answers_str + ' ' + str(s)
    
    images_titles = prepare_image_titles(args.images_file, image_answers_str)

    item_ids, item_reps = prepare_gen_passage_rep_output(args.gen_passage_rep_output)
    gpu_index = construct_passage_faiss_index(args.proj_size, item_reps)

    qrels = prepare_qrels(args.qrels)

    item_id_to_idx = prepare_item_id_to_idx(item_ids)

    qrels_data, qrels_row_idx, qrels_col_idx, qid_to_idx = prepare_qrels_data(qrels, item_id_to_idx)

    qrels_sparse_matrix = scipy.sparse.csr_matrix((qrels_data, (qrels_row_idx, qrels_col_idx)))
    
    return itemid_modalities, passages_dict, tables_dict, images_dict, images_titles, item_ids, item_reps, gpu_index, qrels, item_id_to_idx, qrels_data, qrels_row_idx, qrels_col_idx, qid_to_idx, qrels_sparse_matrix

def prepare_passages(passages_file_path : str, itemid_modalities : list) -> dict:
    """load passages to passages_dict

    Args:
        passages_file_path (str): passages .jsonl file path
        itemid_modalities (list): list storing modalities of each item

    Returns:
        dict: passages_dict (key: itemid, value: passage text)
    """
    passages_dict = {}
    with open(passages_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            passages_dict[line['id']] = line['text']
            itemid_modalities.append('text')
    return passages_dict

def prepare_tables(tables_file_path : str, itemid_modalities : list) -> dict:
    """

    Args:
        tables_file_path (str): _description_
        itemid_modalities (list): _description_

    Returns:
        dict: _description_
    """
    tables_dict = {}
    with open(tables_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            table_context = ''
            for row_data in line['table']['table_rows']:
                for cell in row_data:
                    table_context = table_context + ' ' + cell['text']
            tables_dict[line['id']] = table_context
            itemid_modalities.append('table')
    return tables_dict

def prepare_images(images_file_path : str, images_path : str, itemid_modalities : list) -> dict:
    """load images to images_dict

    Args:
        images_file_path (str): images .jsonl file path
        images_path (str): images folder path
        itemid_modalities (list): list storing modalities of each item

    Returns:
        dict: images_dict (key: itemid, value: image file path)
    """    
    images_dict = {}
    with open(images_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            images_dict[line['id']] = os.path.join(images_path, line['path'])
            # images_titles[line['id']] = line['title']
            itemid_modalities.append('image')
    return images_dict

def prepare_image_titles(images_file_path : str, image_answers_str : str) -> dict:
    """load images info to images_titles

    Args:
        images_file_path (str): images .jsonl file path
        image_answers_str (str): answer str for images

    Returns:
        dict: images_titles (key: itemid, value: image title)
    """
    images_titles = {}
    with open(images_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            images_titles[line['id']] = line['title'] + ' ' + image_answers_str
    return images_titles

def prepare_gen_passage_rep_output(gen_passage_rep_output_path : str) -> Tuple[np.array, np.array]:
    """load gen_passage_rep_output to item_ids and item_reps

    Args:
        gen_passage_rep_output_path (str): .txt file for gen_passage_rep

    Returns:
        Tuple[np.array, np.array]: item_ids, item_reps
    """
    item_ids, item_reps = [], []
    with open(gen_passage_rep_output_path, 'r') as f:
        for line in tqdm(f):
            dic = json.loads(line.strip())
            item_ids.append(dic['id'])
            item_reps.append(dic['rep'])
    return np.asarray(item_ids), np.asarray(item_reps, dtype='float32')

def construct_passage_faiss_index(proj_size : int, item_reps : dict) -> faiss.IndexFlatIP:
    """construct faiss index for passages

    Args:
        proj_size (int): projection size
        item_reps (dict): item representations

    Returns:
        faiss.IndexFlatIP: faiss index
    """
    index = faiss.IndexFlatIP(proj_size)
    index.add(item_reps)
    return index

def prepare_qrels(qrels_file_path : str) -> dict:
    """load qrels to qrels_dict

    Args:
        qrels_file_path (str): qrels .tsv file path

    Returns:
        dict: qrels_dict (key: queryid, value: list of relevant itemids)
    """
    qrels = {}
    with open(qrels_file_path, 'r') as f:
        qrels = json.load(f)
    return qrels

def prepare_item_id_to_idx(item_ids : list) -> dict:
    """load item_ids to item_id_to_idx_dict

    Args:
        item_ids (list): list of item ids

    Returns:
        dict: item_id_to_idx_dict (key: itemid, value: index)
    """
    item_id_to_idx_dict = {}
    for idx, item_id in enumerate(item_ids):
        item_id_to_idx_dict[item_id] = idx
    return item_id_to_idx_dict

def prepare_qrels_data(qrels : dict, item_id_to_idx : dict) -> Tuple[list, list, list, dict]:
    """prepare qrels data for sparse matrix

    Args:
        qrels (dict): query representation
        item_id_to_idx (dict): index mapping

    Returns:
        Tuple[list, list, list, dict]: qrels_data, qrels_row_idx, qrels_col_idx, qid_to_idx
    """    
    qrels_data, qrels_row_idx, qrels_col_idx = [], [], []
    qid_to_idx = {}
    for i, (qid, v) in enumerate(qrels.items()):
        qid_to_idx[qid] = i
        for item_id in v:
            qrels_data.append(1)
            qrels_row_idx.append(i)
            qrels_col_idx.append(item_id_to_idx[item_id])
    qrels_data.append(0)
    qrels_row_idx.append(5752)
    qrels_col_idx.append(285384)
    return qrels_data, qrels_row_idx, qrels_col_idx, qid_to_idx
