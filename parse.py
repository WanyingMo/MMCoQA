import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # arguments shared by the retriever and reader

    parser.add_argument(
        "--train_file",
        default='/home/share/liyongqi/project/MMCoQA/data/MMCoQA_data/final_data/QA pairs/MMCoQA_train.txt',
        type=str,
        required=False,
        help="open retrieval quac json for training."
    )
    parser.add_argument(
        "--dev_file",
        default='/home/share/liyongqi/project/MMCoQA/data/MMCoQA_data/final_data/QA pairs/MMCoQA_dev.txt',
        type=str,
        required=False,
        help="open retrieval quac json for predictions."
    )
    parser.add_argument(
        "--test_file",
        default='/home/share/liyongqi/project/MMCoQA/data/MMCoQA_data/final_data/QA pairs/MMCoQA_test.txt',
        type=str,
        required=False,
        help="open retrieval quac json for predictions."
    )

    parser.add_argument(
        "--passages_file", 
        default='/home/share/liyongqi/project/MMCoQA/data/MMCoQA_data/final_data/multimodal_evidence_collection/texts/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl',
        type=str,
        help="the file contains passages"
    )
    parser.add_argument(
        "--tables_file", 
        default='/home/share/liyongqi/project/MMCoQA/data/MMCoQA_data/final_data/multimodal_evidence_collection/tables/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl',
        type=str,
        help="the file contains passages"
    )
    parser.add_argument(
        "--images_file", 
        default='/home/share/liyongqi/project/MMCoQA/data/MMCoQA_data/final_data/multimodal_evidence_collection/images/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl',
        type=str,
        help="the file contains passages"
    )

    parser.add_argument(
        "--qrels",
        default='dataset/qrels.txt',
        type=str,
        required=False,
        help="qrels to evaluate open retrieval"
    )
    parser.add_argument(
        "--images_path", 
        default="/home/share/liyongqi/project/MMCoQA/data/MMCoQA_data/final_data/multimodal_evidence_collection/images/final_dataset_images/",
        type=str,
        help="the path to images"
    )

    parser.add_argument(
        "--gen_passage_rep_output", 
        default='./retriever_release_test/dev_blocks.txt',
        type=str,
        help="passage representations"
    )
    parser.add_argument(
        "--output_dir",
        default='./release_test',
        type=str,
        required=False,
        help="The output directory where the model checkpoints and predictions will be written."
    )
    parser.add_argument(
        "--load_small",
        default=False,
        type=str2bool,
        required=False,
        help="whether to load just a small portion of data during development"
    )
    parser.add_argument(
        "--num_workers",
        default=2,
        type=int,
        required=False,
        help="number of workers for dataloader"
    )

    parser.add_argument(
        "--global_mode",
        default=True,
        type=str2bool,
        required=False,
        help="maxmize the prob of the true answer given all passages"
    )
    parser.add_argument(
        "--history_num",
        default=1,
        type=int,
        required=False,
        help="number of history turns to use"
    )
    parser.add_argument(
        "--prepend_history_questions",
        default=True,
        type=str2bool,
        required=False,
        help="whether to prepend history questions to the current question"
    )
    parser.add_argument(
        "--prepend_history_answers",
        default=False,
        type=str2bool,
        required=False,
        help="whether to prepend history answers to the current question"
    )

    parser.add_argument(
        "--do_train",
        default=True,
        type=str2bool,
        help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval",
        default=True,
        type=str2bool,
        help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test",
        default=True,
        type=str2bool,
        help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--best_global_step",
        default=12000,
        type=int,
        required=False,
        help="used when only do_test"
    )
    parser.add_argument(
        "--evaluate_during_training",
        default=False,
        type=str2bool,
        help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=str2bool,
        help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--warmup_portion",
        default=0.1,
        type=float,
        help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps "
    )
    parser.add_argument(
        "--verbose_logging",
        action='store_true',
        help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation."
    )

    parser.add_argument(
        '--logging_steps',
        type=int,
        default=10,
        help="Log every X updates steps."
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=4000,
        help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        default=True,
        type=str2bool,
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        type=str2bool,
        help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        '--overwrite_output_dir',
        default=False,
        type=str2bool,
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        '--overwrite_cache',
        action='store_true',
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="random seed for initialization"
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        '--fp16',
        default=False,
        type=str2bool,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." +
        " See details at https://nvidia.github.io/apex/amp.html"
    )
    parser.add_argument(
        '--server_ip',
        type=str,
        default='',
        help="Can be used for distant debugging."
    )
    parser.add_argument(
        '--server_port',
        type=str,
        default='',
        help="Can be used for distant debugging."
    )

    # retriever arguments
    parser.add_argument(
        "--retriever_config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--retriever_model_type",
        default='albert',
        type=str,
        required=False,
        help="retriever model type"
    )
    parser.add_argument(
        "--retriever_model_name_or_path",
        default='albert-base-v2',
        type=str,
        required=False,
        help="retriever model name"
    )
    parser.add_argument(
        "--retriever_tokenizer_name",
        default="albert-base-v2",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        "--retriever_cache_dir",
        default="../huggingface_cache/albert-base-v2/",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3"
    )
    parser.add_argument(
        "--retrieve_checkpoint",
        default='./retriever_release_test/checkpoint-5061',
        type=str,
        help="generate query/passage representations with this checkpoint"
    )
    parser.add_argument(
        "--retrieve_tokenizer_dir",
        default='./retriever_release_test',
        type=str,
        help="dir that contains tokenizer files"
    )

    parser.add_argument(
        "--given_query",
        default=True,
        type=str2bool,
        help="Whether query is given."
    )
    parser.add_argument(
        "--given_passage",
        default=False,
        type=str2bool,
        help="Whether passage is given. Passages are not given when jointly train"
    )
    parser.add_argument(
        "--is_pretraining",
        default=False,
        type=str2bool,
        help="Whether is pretraining. We fine tune the query encoder in retriever"
    )
    parser.add_argument(
        "--include_first_for_retriever",
        default=True,
        type=str2bool,
        help="include the first question in a dialog in addition to history_num for retriever (not reader)"
    )
    # parser.add_argument("--only_positive_passage", default=True, type=str2bool,
    #                     help="we only pass the positive passages, the rest of the passges in the batch are considered as negatives")
    parser.add_argument(
        "--retriever_query_max_seq_length",
        default=30,
        type=int,
        help="The maximum input sequence length of query."
    )
    parser.add_argument(
        "--retriever_passage_max_seq_length",
        default=128,
        type=int,
        help="The maximum input sequence length of passage (384 + [CLS] + [SEP])."
    )
    parser.add_argument(
        "--proj_size",
        default=128,
        type=int,
        help="The size of the query/passage rep after projection of [CLS] rep."
    )
    parser.add_argument(
        "--top_k_for_retriever",
        default=2000,
        type=int,
        help="retrieve top k passages for a query, these passages will be used to update the query encoder"
    )
    parser.add_argument(
        "--use_retriever_prob",
        default=True,
        type=str2bool,
        help="include albert retriever probs in final answer ranking"
    )

    # reader arguments
    parser.add_argument(
        "--reader_config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--reader_model_name_or_path",
        default='bert-base-uncased',
        type=str,
        required=False,
        help="reader model name"
    )
    parser.add_argument(
        "--reader_model_type",
        default='bert',
        type=str,
        required=False,
        help="reader model type"
    )
    parser.add_argument(
        "--reader_tokenizer_name",
        default="bert-base-uncased",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        "--reader_cache_dir",
        default="../huggingface_cache/bert-base-uncased/",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3"
    )
    parser.add_argument(
        "--reader_max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. " +
        "Sequences longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--doc_stride",
        default=384,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks."
    )
    parser.add_argument(
        '--version_2_with_negative',
        default=False,
        type=str2bool,
        required=False,
        help='If true, the SQuAD examples contain some that do not have an answer.'
    )
    parser.add_argument(
        '--null_score_diff_threshold',
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null."
    )
    parser.add_argument(
        "--reader_max_query_length",
        default=125,
        type=int,
        help="The maximum number of tokens for the question. " +
        "Questions longer than this will be truncated to this length."
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file."
    )
    parser.add_argument(
        "--max_answer_length",
        default=40,
        type=int,
        help="The maximum length of an answer that can be generated. " +
        "This is needed because the start and end predictions are not conditioned on one another."
    )
    parser.add_argument(
        "--qa_loss_factor",
        default=1.0,
        type=float,
        help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss"
    )
    parser.add_argument(
        "--retrieval_loss_factor",
        default=1.0,
        type=float,
        help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss"
    )
    parser.add_argument(
        "--top_k_for_reader",
        default=10,
        type=int,
        help="update the reader with top k passages"
    )
    parser.add_argument(
        "--use_rerank_prob",
        default=True,
        type=str2bool,
        help="include rerank probs in final answer ranking"
    )
    return parser