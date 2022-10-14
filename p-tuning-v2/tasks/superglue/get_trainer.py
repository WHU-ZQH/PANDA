import logging
import os
import random
import sys
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.superglue.dataset import SuperGlueDataset
from training.trainer_base import BaseTrainer
from training.trainer_exp import ExponentialTrainer

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = SuperGlueDataset(tokenizer, data_args, training_args)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.multiple_choice:
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

    if not dataset.multiple_choice:
        model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    else:
        model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config, fix_bert=True)

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

    # Initialize our Trainer  model.bert.encoder.layer[0].intermediate.regi
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        test_key=dataset.test_key,
        model_args=model_args
    )


    return trainer, None
