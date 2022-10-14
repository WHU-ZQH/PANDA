import logging
import os
import random
import sys
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from tasks.qa.dataset import SQuAD
from training.trainer_qa import QuestionAnsweringTrainer
from model.utils import get_model, TaskType

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, qa_args = args

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        revision=model_args.model_revision,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
    )

    model = get_model(model_args, TaskType.QUESTION_ANSWERING, config, fix_bert=True)

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

    dataset = SQuAD(tokenizer, data_args, training_args, qa_args)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        eval_examples=dataset.eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        post_process_function=dataset.post_processing_function,
        compute_metrics=dataset.compute_metrics,
        model_args=model_args
    )

    return trainer, dataset.predict_dataset


