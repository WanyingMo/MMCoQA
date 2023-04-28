# MMCoQA

Reproduce and modify the paper: [MMCoQA: Conversational Question Answering over Text, Tables, and Images](https://aclanthology.org/2022.acl-long.290) (Li et al., ACL 2022)

Command to run the inference:
```
sh scripts/test.sh
```

## Project structure

```
.
├── dataset
|   ├── final_dataset_images
|   |   └── ... .png/.jpg
|   ├── MMCoQA_dev.txt
|   ├── MMCoQA_test.txt
|   ├── MMCoQA_train.txt
|   ├── multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl
|   ├── multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl
|   ├── multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl
|   └── qrels.txt
├── docker
|   ├── build.sh
|   ├── Dockerfile
|   └── run.sh
├── release_test
├── retriever_release_test
├── scripts
|   └── test.sh
└── ...
```

## Setup

Make sure you are in the project root directory

### Build docker image

`sh docker/build.sh`

### Create and run the container

`sh docker/run.sh`

### Install required python packages

`pip install -r requirements.txt`

## Required Python packages
```
torch==2.0.0
pytrec-eval==0.5
faiss-gpu==1.7.2
transformers==2.3.0
tensorboard==2.12.2
tqdm==4.64.1
numpy==1.23.5
scipy==1.10.1
opencv-python==4.7.0.72
```

Required datasets and checkpoints should follow this website: [link](https://github.com/liyongqi67/MMCoQA)
