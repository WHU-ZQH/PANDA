import logging
import os
import random
import torch
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.glue.dataset import GlueDataset
from training.trainer_base import BaseTrainer

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = GlueDataset(tokenizer, data_args, training_args)

    if not dataset.is_regression:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )

    model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

    if model_args.prompt_transfer == 1:
        source_dict = torch.load(model_args.source_prompt, map_location='cuda')
        model_dict = model.state_dict()
        initialized_dict = {k: v for k, v in source_dict.items() if (k in model_dict) and ('classifier' not in k)}
        model_dict.update(initialized_dict)
        model.load_state_dict(model_dict)

    # if model_args.prompt_transfer == 2:
    #     source_dict = torch.load(model_args.target_prompt, map_location='cuda')
    #     model_dict = model.state_dict()
    #     initialized_dict = {k: v for k, v in source_dict.items() if (k in model_dict) and ('classifier' not in k)}
    #     model_dict.update(initialized_dict)
    #     model.load_state_dict(model_dict)

    test_key = "accuracy"
    if data_args.dataset_name == 'cola':
        test_key = 'matthews_correlation'
    elif data_args.dataset_name == 'stsb':
        test_key = 'combined_score'

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        test_key=test_key,
        model_args=model_args
    )

    return trainer, None