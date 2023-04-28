from copy import copy
import glob
import json
import logging
import os
import random

import pytrec_eval
import torch
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, AlbertConfig, AlbertTokenizer
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from retriever_utils import RetrieverDataset
from modeling import Pipeline, BertForOrconvqaGlobal, BertForRetrieverOnlyPositivePassage,AlbertForRetrieverOnlyPositivePassage
from parse import init_parser
from prepare_data import prepare_dataset
from pipeline import train, evaluate

def set_seed(seed : int, n_gpu : int):
    """set random seed

    Args:
        seed (int): random seed
        n_gpu (int): gpu num
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def build_train_model(
        args : dict,
        retriever_model_class : AlbertForRetrieverOnlyPositivePassage,
        reader_config_class : BertConfig, reader_model_class : BertForOrconvqaGlobal
) -> Pipeline:
    """build model for training

    Args:
        args (dict): 
        retriever_model_class (AlbertForRetrieverOnlyPositivePassage): retriever model class
        reader_config_class (BertConfig): reader config class
        reader_model_class (BertForOrconvqaGlobal): reader model class

    Returns:
        Pipeline: model for training
    """
    
    model = Pipeline()

    # load pretrained retriever
    # retriever_tokenizer = retriever_tokenizer_class.from_pretrained(args.retrieve_tokenizer_dir)
    model.retriever = retriever_model_class.from_pretrained(args.retrieve_checkpoint, force_download=True)
    # do not need and do not tune passage encoder
    model.retriever.passage_encoder = None
    model.retriever.passage_proj = None
    model.retriever.image_encoder = None
    model.retriever.image_proj = None

    reader_config = reader_config_class.from_pretrained(
        args.reader_config_name if args.reader_config_name else args.reader_model_name_or_path,
        cache_dir=args.reader_cache_dir if args.reader_cache_dir else None
    )
    reader_config.num_qa_labels = 2
    # this not used for BertForOrconvqaGlobal
    reader_config.num_retrieval_labels = 2
    reader_config.qa_loss_factor = args.qa_loss_factor
    reader_config.retrieval_loss_factor = args.retrieval_loss_factor
    reader_config.proj_size = args.proj_size

    model.reader = reader_model_class.from_pretrained(
        args.reader_model_name_or_path,
        from_tf=bool('.ckpt' in args.reader_model_name_or_path),
        config=reader_config,
        cache_dir=args.reader_cache_dir if args.reader_cache_dir else None
    )

    return model

def build_test_model(
        retriever_model_class : AlbertForRetrieverOnlyPositivePassage,
        reader_model_class : BertForOrconvqaGlobal,
        checkpoint : str
) -> Pipeline:
    """build model for test

    Args:
        retriever_model_class (AlbertForRetrieverOnlyPositivePassage): retriever model class
        reader_model_class (BertForOrconvqaGlobal): reader model class
        checkpoint (str): path of checkpoint

    Returns:
        Pipeline: model for test
    """

    model = Pipeline()

    model.retriever = retriever_model_class.from_pretrained(
        os.path.join(checkpoint, 'retriever'),
        force_download=True
    )
    model.retriever.passage_encoder = None
    model.retriever.passage_proj = None
    model.retriever.image_encoder = None
    model.retriever.image_proj = None
    
    model.reader = reader_model_class.from_pretrained(
        os.path.join(checkpoint, 'reader'),
        force_download=True
    )

    return model

if __name__ == '__main__':

    logger = logging.getLogger(__name__)

    ALL_MODELS = list(BertConfig.pretrained_config_archive_map.keys())

    MODEL_CLASSES = {
        'reader': (BertConfig, BertForOrconvqaGlobal, BertTokenizer),
        'retriever': (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer)
    }

    parser = init_parser()
    args, unknown = parser.parse_known_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
    args.retriever_tokenizer_dir = os.path.join(args.output_dir, 'retriever')
    args.reader_tokenizer_dir = os.path.join(args.output_dir, 'reader')

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
        # torch.cuda.set_device(0)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning(
        f"Process rank: {args.local_rank}, " +
        f"device: {device}, " +
        f"n_gpu: {args.n_gpu}, " +
        f"distributed training: {bool(args.local_rank != -1)}, " +
        f"16-bits training: {args.fp16}"
    )

    # Set seed
    set_seed(args.seed, args.n_gpu)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # load model config and build tokenizer
    retriever_config_class, retriever_model_class, retriever_tokenizer_class = MODEL_CLASSES['retriever']
    reader_config_class, reader_model_class, reader_tokenizer_class = MODEL_CLASSES['reader']

    retriever_tokenizer = retriever_tokenizer_class.from_pretrained(args.retrieve_tokenizer_dir)
    reader_tokenizer = reader_tokenizer_class.from_pretrained(
        args.reader_tokenizer_name if args.reader_tokenizer_name else args.reader_model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.reader_cache_dir if args.reader_cache_dir else None
    )


    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    logger.info(f"Training/evaluation parameters {args}")

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    logger.info(f'loading data...')

    itemid_modalities, passages_dict, tables_dict, images_dict, images_titles, item_ids, item_reps, gpu_index, qrels, item_id_to_idx, qrels_data, qrels_row_idx, qrels_col_idx, qid_to_idx, qrels_sparse_matrix = prepare_dataset(args)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg', 'set_recall'})

    retriever_tokenizer.save_pretrained(args.retriever_tokenizer_dir)
    reader_tokenizer.save_pretrained(args.reader_tokenizer_dir)


    # Training
    if args.do_train:
        train_dataset = RetrieverDataset(
            args.train_file, retriever_tokenizer,
            args.load_small, args.history_num,
            query_max_seq_length=args.retriever_query_max_seq_length,
            is_pretraining=args.is_pretraining,
            prepend_history_questions=args.prepend_history_questions,
            prepend_history_answers=args.prepend_history_answers,
            given_query=True,
            given_passage=False,
            include_first_for_retriever=args.include_first_for_retriever
        )

        model = build_train_model(
            args,
            retriever_model_class,
            reader_config_class, reader_model_class
        )

        model.to(args.device)

        global_step, tr_loss = train(
            args,
            logger, model, evaluator,
            retriever_tokenizer, reader_tokenizer,
            train_dataset,
            itemid_modalities, passages_dict, tables_dict, images_dict, images_titles,
            qid_to_idx, item_ids, item_id_to_idx, item_reps,
            qrels, qrels_sparse_matrix, gpu_index
        )
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")


    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        max_f1 = 0.0
        best_metrics = {}

        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = sorted(
                list(
                    os.path.dirname(os.path.dirname(c)) for c in glob.glob(args.output_dir + '/*/retriever/' + WEIGHTS_NAME, recursive=False)
                )
            )
        logger.info(f"Evaluate the following checkpoints: {checkpoints}")

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoint) > 1 else ""
            print(global_step, 'global_step')
            
            model = build_test_model(retriever_model_class, reader_model_class, checkpoint)

            model.to(args.device)

            result = evaluate(
                args, logger, model, evaluator,
                retriever_tokenizer, reader_tokenizer,
                itemid_modalities, passages_dict, tables_dict, images_dict, images_titles,
                qid_to_idx, item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix, gpu_index,
                prefix=global_step
            )

            if result['f1'] > max_f1:
                max_f1 = result['f1']
                best_metrics = copy(result)
                best_metrics['global_step'] = global_step

            for key, value in result.items():
                tb_writer.add_scalar(f'eval_{key}', value, global_step)

            result = dict(
                (k + (f'_{global_step}' if global_step else ''), v) for k, v in result.items()
            )
            results.update(result)

        best_metrics_file = os.path.join(args.output_dir, 'predictions', 'best_metrics.json')
        with open(best_metrics_file, 'w') as fout:
            json.dump(best_metrics, fout)

        all_results_file = os.path.join(args.output_dir, 'predictions', 'all_results.json')
        with open(all_results_file, 'w') as fout:
            json.dump(results, fout)

        logger.info(f"Results: {results}")
        logger.info(f"best metrics: {best_metrics}")


    # Test
    if args.do_test and args.local_rank in [-1, 0]:
        best_global_step = best_metrics['global_step'] if args.do_eval else args.best_global_step
        best_checkpoint = os.path.join(args.output_dir, f'checkpoint-{best_global_step}')
        logger.info(f"Test the best checkpoint: {best_checkpoint}")

        model = build_test_model(retriever_model_class, reader_model_class, best_checkpoint)

        model.to(args.device)

        result = evaluate(
            args, logger, model, evaluator,
            retriever_tokenizer, reader_tokenizer,
            itemid_modalities, passages_dict, tables_dict, images_dict, images_titles,
            qid_to_idx, item_ids, item_id_to_idx, item_reps, qrels, qrels_sparse_matrix, gpu_index,
            prefix='test'
        )

        test_metrics_file = os.path.join(args.output_dir, 'predictions', 'test_metrics.json')
        with open(test_metrics_file, 'w') as f:
            json.dump(result, f)

        logger.info(f"Test Result: {result}")
