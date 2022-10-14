import logging
import os
from typing import Dict, OrderedDict
import torch
import torch.nn as nn
import time
import copy
import math
from transformers.trainer_pt_utils import nested_detach, get_parameter_names
from transformers.trainer_utils import ShardedDDPOption, speed_metrics
# from transformers.file_utils import is_sagemaker_mp_enabled
from transformers import Trainer
from transformers.optimization import Adafactor, AdamW, get_scheduler
# from fairscale.optim import OSS

logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)

class BaseTrainer(Trainer):
    def __init__(self, *args, predict_dataset = None, test_key = "accuracy", model_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_dataset = predict_dataset
        self.test_key = test_key
        self.model_args = model_args
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.test_key}": 0,
        })
        if self.model_args.prompt_transfer == 2:
            self.source_model = copy.deepcopy(kwargs["model"])
            
            model_dict = self.source_model.state_dict()
            source_dict = torch.load(self.model_args.source_prompt, map_location='cuda')
            # if self.model_args.beta==0.1:
            #     initialized_dict = {k: v for k, v in source_dict.items() if (k in model_dict) and ('classifier' not in k)}
            # else:
            initialized_dict = {k: v for k, v in source_dict.items() if k in model_dict}
            model_dict.update(initialized_dict)
            self.source_model.load_state_dict(model_dict)
            self.source_model=self._wrap_model(self.source_model)
            self.source_model.zero_grad()  
            # for param in self.source_model.parameters():
            #     param.requires_grad = False
    
        # self.outputs=[[] for _ in range(kwargs["model"].n_layer)]
        # def save_ppt_outputs1_hook(n):
        #     def fn(_,__,output):
        #         self.outputs[n].append(output.detach().to("cpu"))
        #     return fn

        # for n in range(kwargs["model"].n_layer):
        #     kwargs["model"].bert.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:
            eval_metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, eval_metrics)

            if eval_metrics["eval_"+self.test_key] > self.best_metrics["best_eval_"+self.test_key]:
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_"+self.test_key] = eval_metrics["eval_"+self.test_key]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(dataset, metric_key_prefix="test")
                            self.best_metrics[f"best_test_{dataset_name}_{self.test_key}"] = test_metrics["test_"+self.test_key]
                    else:
                        _, _, test_metrics = self.predict(self.predict_dataset, metric_key_prefix="test")
                        self.best_metrics["best_test_"+self.test_key] = test_metrics["test_"+self.test_key]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

            if (eval_metrics["eval_"+self.test_key] == self.best_metrics["best_eval_"+self.test_key]) and (self.control.should_save):
                self._save_checkpoint(model, trial, metrics=eval_metrics)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            # if self.model_args.prompt_transfer == 2:
            #     optimizer_grouped_parameters.append({
            #         "params": [p for n, p in self.source_model.named_parameters() if n in decay_parameters],
            #         "weight_decay": self.args.weight_decay,})
            #     optimizer_grouped_parameters.append({
            #         "params": [p for n, p in self.source_model.named_parameters() if n not in decay_parameters],
            #         "weight_decay": 0.0,})

            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def get_beta(self):
        warmup_epoch=int(self.state.num_train_epochs*0.1)
        other_epoch=self.state.num_train_epochs-warmup_epoch
        if self.state.epoch < warmup_epoch:
            beta=self.model_args.beta*(self.state.epoch/warmup_epoch)
        else:
            now_epoch=self.state.epoch-warmup_epoch
            beta=self.model_args.beta*(1-now_epoch/other_epoch)
        return beta

    def get_beta2(self):
        warmup_epoch=int(self.state.num_train_epochs*0.1)
        other_epoch=self.state.num_train_epochs-warmup_epoch
        if self.state.epoch < warmup_epoch:
            beta=self.model_args.beta
        else:
            now_epoch=self.state.epoch-warmup_epoch
            beta=self.model_args.beta*(1-now_epoch/other_epoch)
        return beta

    def training_step(self, model, inputs):
            model.train()
            inputs = self._prepare_inputs(inputs)
            if self.model_args.prompt_transfer == 2:
                self.source_model.train()
                loss, loss_mse = self.compute_loss(model, inputs)
                # if self.args.n_gpu > 1:
                #     loss2 = loss2.mean() 
                # if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                #     loss2 = loss2 / self.args.gradient_accumulation_steps
            else:
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean() 

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss = loss / self.args.gradient_accumulation_steps

            if self.model_args.prompt_transfer == 2:
                beta=self.get_beta2()
                # if self.state.epoch > int(self.state.num_train_epochs*0.0):
                loss = loss+0.1*self.model_args.beta*loss_mse
                # loss = loss+beta*loss_mse

            if self.use_amp:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()
            if self.model_args.prompt_transfer == 2:
                self.source_model.zero_grad()  
            return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # model_dict_student = self.source_model.state_dict().copy()
        # model_dict_teacher = model.state_dict().copy()
        # model_dict_student = {k: v for k, v in model_dict_student.items()}
        # model_dict_teacher = {k: v for k, v in model_dict_teacher.items()}
        # model_dict = {}
        # for k in model_dict_student.keys():
        #     parameters = 0.99 * model_dict_teacher.get(k) + 0.01 * model_dict_student.get(k)
        #     model_dict[k] = parameters
        # model.load_state_dict(model_dict)

        outputs = model(**inputs)

        if self.model_args.prompt_transfer == 2:
            outputs2= self.source_model(**inputs)
            logit, logit2=outputs["logits"], outputs2["logits"]
            # logit.requires_grad_()
            # logit2.requires_grad_()
            loss_mse=torch.nn.functional.mse_loss(logit, logit2)/logit.shape[0]   ## 1:/logit.shape[0], 2:/

            # logit=outputs["logits"]
            # hidden1, hidden2=outputs["hidden_states"], outputs2["hidden_states"]
            # hidden1.requires_grad_()
            # hidden2.requires_grad_()
            # loss_mse=torch.nn.functional.mse_loss(hidden1,hidden2)/logit.shape[0]

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
                # loss2 = self.label_smoother(outputs, labels)
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                # loss2 = outputs2["loss"] if isinstance(outputs2, dict) else outputs2[0]

            return (loss, outputs) if return_outputs else (loss, loss_mse)
        else:
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss

    # def evaluate(
    #     self,
    #     eval_dataset=None,
    #     ignore_keys= None,
    #     metric_key_prefix: str = "eval",
    # ) -> Dict[str, float]:
    #     """
    #     Run evaluation and returns metrics.

    #     The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
    #     (pass it to the init :obj:`compute_metrics` argument).

    #     You can also subclass and override this method to inject custom behavior.

    #     Args:
    #         eval_dataset (:obj:`Dataset`, `optional`):
    #             Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
    #             columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
    #             :obj:`__len__` method.
    #         ignore_keys (:obj:`Lst[str]`, `optional`):
    #             A list of keys in the output of your model (if it is a dictionary) that should be ignored when
    #             gathering predictions.
    #         metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
    #             An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
    #             "eval_bleu" if the prefix is "eval" (default)

    #     Returns:
    #         A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
    #         dictionary also contains the epoch number which comes from the training state.
    #     """
    #     # memory metrics - must set up as early as possible
    #     self._memory_tracker.start()

    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     start_time = time.time()

    #     eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
    #     output = eval_loop(
    #         eval_dataloader,
    #         description="Evaluation",
    #         # No point gathering the predictions if there are no metrics, otherwise we defer to
    #         # self.args.prediction_loss_only
    #         prediction_loss_only=True if self.compute_metrics is None else None,
    #         ignore_keys=ignore_keys,
    #         metric_key_prefix=metric_key_prefix,
    #     )

    #     for k in range(self.model.n_layer):
    #         self.outputs[k]=torch.cat(self.outputs[k])

    #     self.outputs=torch.stack(self.outputs)
    #     self.outputs=self.outputs[:,:,:1,:]

    #     name=""
    #     if self.model.config.name_or_path=='bert-base-cased':
    #         name="bert-base"
    #     elif "small" in self.model.config.name_or_path:
    #         name="bert-small"
    #     elif "medium" in self.model.config.name_or_path:
    #         name="bert-medium"
    #     elif "tiny" in self.model.config.name_or_path:
    #         name="bert-tiny"
    #     else :
    #         name="bert-large"
    #     torch.save(self.outputs, 
    #     "p-tuning-v2/{}_similarity/model_tuning/{}_on.pt".format(name,self.eval_dataset.config_name))

    #     total_batch_size = self.args.eval_batch_size * self.args.world_size

    #     output.metrics.update(
    #         speed_metrics(
    #             metric_key_prefix,
    #             start_time,
    #             num_samples=output.num_samples,
    #             num_steps=math.ceil(output.num_samples / total_batch_size),
    #         )
    #     )

    #     self.log(output.metrics)

    #     self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

    #     self._memory_tracker.stop_and_update_metrics(output.metrics)

    #     return output.metrics
