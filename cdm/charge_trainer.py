import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.trainers.base_trainer import BaseTrainer
from ocpmodels.common.utils import pyg2_data_transform

from ocpmodels.common.data_parallel import (
    BalancedBatchSampler,
    OCPDataParallel,
    ParallelCollater,
)

from ocpmodels.modules.loss import AtomwiseL2Loss, DDPLoss, L2MAELoss
from ocpmodels.modules.evaluator import *

from cdm.utils.probe_graph import ProbeGraphAdder
from cdm import models


@registry.register_trainer("charge")
class ChargeTrainer(BaseTrainer):
    """
    Trainer class for charge density prediction task.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_is2re <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_.


    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_hpo=False,
        print_every=100,
        log_every=100,
        seed=None,
        logger="wandb",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        name=None,
        trainer = 'charge',
    ):
        
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name='s2ef',
            slurm=slurm,
            noddp=noddp,
        )
    
        self.evaluator = ChargeEvaluator()
        self.name = 'charge'
        self.log_every = log_every
    
    def load_loss(self):
        
        self.loss_fn = {}
        self.loss_fn['charge'] = self.config['optim'].get('loss_charge', 'mae')
        
        for loss, loss_name in self.loss_fn.items():
            if loss_name in ['l1', 'mae']:
                self.loss_fn[loss] = torch.nn.L1Loss()
            elif loss_name in ['l2', 'mse']:
                self.loss_fn[loss] = torch.nn.MSELoss()
            elif loss_name == 'l2mae':
                self.loss_fn[loss] = L2MAELoss()
            elif loss_name == 'atomwisel2':
                self.loss_fn[loss] = AtomwiseL2Loss()
            elif loss_name == 'normed_mae':
                self.loss_fn[loss] = NormedMAELoss()
            else:
                raise NotImplementedError(
                    f'Unknown loss function name: {loss_name}'
                )
            
            self.loss_fn[loss] = DDPLoss(self.loss_fn[loss], reduction='sum')
        

    def load_task(self):
        logging.info(f"Loading dataset: {self.config['task']['dataset']}")
        self.num_targets = 1

    @torch.no_grad()
    def predict(
        self, loader, per_image=True, results_file=None, disable_tqdm=False
    ):
        if distutils.is_master() and not disable_tqdm:
            logging.info('Predicting on test.')
        assert isinstance(
            loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(loader, torch_geometric.data.Batch):
            loader = [[loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and 'target' in self.normalizers:
            self.normalizers['target'].to(self.device)
        predictions = {'id': [], 'charge': []}

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc='device {}'.format(rank),
            disable=disable_tqdm,
        ):
            
            if hasattr(batch[0], 'probe_data'):
                for subbatch in batch:
                    subbatch.probe_data = [pyg2_data_transform(x) for x in subbatch.probe_data]
                    subbatch.probe_data = Batch.from_data_list(subbatch.probe_data)

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            if self.normalizers is not None and 'target' in self.normalizers:
                out['charge'] = self.normalizers['target'].denorm(
                    out['charge']
                )

            if per_image:
                predictions['id'].extend(
                    [str(i) for i in batch[0].sid.tolist()]
                )
                predictions['charge'].extend(out['charge'].tolist())
            else:
                predictions['charge'] = out['charge'].detach()
                return predictions

        self.save_results(predictions, results_file, keys=['charge'])

        if self.ema:
            self.ema.restore()

        return predictions

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config['optim'].get(
            'eval_every', len(self.train_loader)
        )
        primary_metric = self.config['task'].get(
            'primary_metric', self.evaluator.task_primary_metric[self.name]
        )
        self.best_val_metric = 1e9

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config['optim']['max_epochs']
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.

                batch = next(train_loader_iter)
                
                if hasattr(batch[0], 'probe_data'):
                    for subbatch in batch:
                        subbatch.probe_data = [pyg2_data_transform(x) for x in subbatch.probe_data]
                        subbatch.probe_data = Batch.from_data_list(subbatch.probe_data)
                
                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                
                self._backward(loss)
                
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    metrics={},
                )

                # Log metrics.
                if (self.step % self.log_every == 0) or (self.step % self.config['cmd']['print_every'] == 0):
                    log_dict = {k: self.metrics[k]['metric'] for k in self.metrics}
                    log_dict.update(
                        {
                            'lr': self.scheduler.get_lr(),
                            'epoch': self.epoch,
                            'step': self.step,
                        }
                    )
                    if (
                        self.step % self.config['cmd']['print_every'] == 0
                        and distutils.is_master()
                        and not self.is_hpo
                    ):
                        log_str = [
                            '{}: {:.2e}'.format(k, v) for k, v in log_dict.items()
                        ]
                        print(', '.join(log_str))
                        self.metrics = {}

                    if self.logger is not None:
                        self.logger.log(
                            log_dict,
                            step=self.step,
                            split='train',
                        )

                # Evaluate on val set after every `eval_every` iterations.
                if self.step % eval_every == 0:
                    self.save(
                        checkpoint_file='checkpoint.pt', training_state=True
                    )

                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split='val',
                            disable_tqdm=disable_eval_tqdm,
                        )
                        if (
                            val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]['metric']
                            < self.best_val_metric
                        ):
                            self.best_val_metric = val_metrics[
                                self.evaluator.task_primary_metric[self.name]
                            ]['metric']
                            self.save(
                                metrics=val_metrics,
                                checkpoint_file='best_checkpoint.pt',
                                training_state=False,
                            )
                            if self.test_loader is not None:
                                self.predict(
                                    self.test_loader,
                                    results_file='predictions',
                                    disable_tqdm=False,
                                )

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                if self.scheduler.scheduler_type == 'ReduceLROnPlateau':
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]['metric'],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

        self.train_dataset.close_db()
        if self.config.get('val_dataset', False):
            self.val_dataset.close_db()
        if self.config.get('test_dataset', False):
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        output = self.model(batch_list)

        if output.shape[-1] == 1:
            output = output.view(-1)

        return {
            'charge': output,
        }

    def _compute_loss(self, out, batch_list):
        
        charge_target = torch.cat(
            [batch.probe_data.target for batch in batch_list], dim=0
        )

        if self.normalizer.get('normalize_labels', False):
            target_normed = self.normalizers['target'].norm(charge_target)
        else:
            target_normed = charge_target

        loss = self.loss_fn['charge'](out['charge'], target_normed)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        charge_target = torch.cat(
             [batch.probe_data.target for batch in batch_list], dim=0
        )

        if self.normalizer.get('normalize_labels', False):
            out['charge'] = self.normalizers['target'].denorm(out['charge'])
        
        metrics = evaluator.eval(
            out,
            {'charge': charge_target},
            prev_metrics=metrics,
        )

        return metrics
    
    def load_model(self):
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        # TODO: depreicated, remove.
        bond_feat_dim = None
        bond_feat_dim = self.config['model_attributes'].get(
            'num_gaussians', 50
        )

        loader = self.train_loader or self.val_loader or self.test_loader
        
        self.model = registry.get_model_class(self.config['model'])(
            **self.config['model_attributes'],
        ).to(self.device)

        if distutils.is_master():
            logging.info(
                f'Loaded {self.model.__class__.__name__} with '
                f'{self.model.num_params} parameters.'
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config['noddp']:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device]
            )
    @torch.no_grad()
    def validate(self, split='val', disable_tqdm=False):
        if distutils.is_master():
            logging.info(f'Evaluating on {split}.')
        if self.is_hpo:
            disable_tqdm = True

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator, metrics = ChargeEvaluator(task='charge'), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split =='val' else self.test_loader
        
        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc='device {}'.format(rank),
            disable=disable_tqdm,
        ):
            if hasattr(batch[0], 'probe_data'):
                for subbatch in batch:
                    subbatch.probe_data = [pyg2_data_transform(x) for x in subbatch.probe_data]
                    subbatch.probe_data = Batch.from_data_list(subbatch.probe_data)

            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update('loss', loss.item(), metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                'total': distutils.all_reduce(
                    metrics[k]['total'], average=False, device=self.device
                ),
                'numel': distutils.all_reduce(
                    metrics[k]['numel'], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]['metric'] = (
                aggregated_metrics[k]['total'] / aggregated_metrics[k]['numel']
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]['metric'] for k in metrics}
        log_dict.update({'epoch': self.epoch})
        if distutils.is_master():
            log_str = ['{}: {:.4f}'.format(k, v) for k, v in log_dict.items()]
            logging.info(', '.join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics
    
    def load_datasets(self):
        
        self.config['optim']['batch_size'] = 1
        self.config['optim']['eval_batch_size'] = 1
        
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            self.config['model_attributes'].get('otf_graph', False),
        )

        self.train_loader = self.val_loader = self.test_loader = None

        if self.config.get('dataset', None):
            self.train_dataset = registry.get_dataset_class(
                self.config['task']['dataset']
            )(self.config['dataset'])
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config['optim']['batch_size'],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

            if self.config.get('val_dataset', None):
                self.val_dataset = registry.get_dataset_class(
                    self.config['task']['dataset']
                )(self.config['val_dataset'])
                self.val_sampler = self.get_sampler(
                    self.val_dataset,
                    self.config['optim'].get(
                        'eval_batch_size', self.config['optim']['batch_size']
                    ),
                    shuffle=False,
                )
                self.val_loader = self.get_dataloader(
                    self.val_dataset,
                    self.val_sampler,
                )

            if self.config.get('test_dataset', None):
                self.test_dataset = registry.get_dataset_class(
                    self.config['task']['dataset']
                )(self.config['test_dataset'])
                self.test_sampler = self.get_sampler(
                    self.test_dataset,
                    self.config['optim'].get(
                        'eval_batch_size', self.config['optim']['batch_size']
                    ),
                    shuffle=False,
                )
                self.test_loader = self.get_dataloader(
                    self.test_dataset,
                    self.test_sampler,
                )

        # Normalizer for the dataset.
        # Compute mean, std of training set labels.
        self.normalizers = {}
        if self.normalizer.get('normalize_labels', False):
            if 'target_mean' in self.normalizer:
                self.normalizers['target'] = Normalizer(
                    mean=self.normalizer['target_mean'],
                    std=self.normalizer['target_std'],
                    device=self.device,
                )
            else:
                self.normalizers["target"] = Normalizer(
                    tensor=self.train_loader.dataset.data.y[
                        self.train_loader.dataset.__indices__
                    ],
                    device=self.device,
                )
                
    def get_dataloader(self, dataset, sampler):
            loader = DataLoader(
                dataset,
                collate_fn=self.parallel_collater,
                num_workers=self.config["optim"]["num_workers"],
                pin_memory=True,
                batch_sampler=sampler,
                prefetch_factor = 6,
            )
            return loader


class ChargeEvaluator(Evaluator):
    def __init__(self, task = 'charge'):
        self.task = 'charge'
        
        self.task_metrics['charge'] = [
            'norm_charge_mae',
            'norm_charge_rmse',
            'charge_mae',
            'charge_mse',
            'true_density',
            'total_charge_ratio',
        ]
        
        self.task_attributes['charge'] = ['charge']
        self.task_primary_metric['charge'] = 'norm_charge_mae'
        
        self.metric_fn = self.task_metrics[task]
        
    def eval(self, prediction, target, prev_metrics={}):
        for attr in self.task_attributes[self.task]:
            assert attr in prediction
            assert attr in target
            assert prediction[attr].shape == target[attr].shape

        metrics = prev_metrics

        for fn in self.task_metrics[self.task]:
            res = eval(fn)(prediction, target)
            metrics = self.update(fn, res, metrics)

        return metrics
    
class NormedMAELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        prediction,
        target,
    ):
        return torch.sum(torch.abs(prediction - target)) \
             / torch.sum(torch.abs(target))
    
def absolute_error(prediction, target):
    error = torch.abs(prediction - target)
    return {
        'metric': (torch.mean(error)).item(),
        'total': torch.sum(error).item(),
        'numel': prediction.numel(),
    }

def squared_error(prediction, target):
    error = torch.abs(prediction - target) **2
    return {
        'metric': (torch.mean(error)).item(),
        'total': torch.sum(error).item(),
        'numel': prediction.numel(),
    }

def charge_mae(prediction, target):
    return absolute_error(prediction['charge'], target['charge'])

def charge_mse(prediction, target):
    return squared_error(prediction['charge'].float(), target['charge'].float())


def total_charge_ratio(prediction, target):
    error = torch.sum(prediction['charge']) / torch.sum(target['charge'])
    return {
        'metric': torch.mean(error).item(),
        'total': torch.sum(error).item(),
        'numel': error.numel()
    }

def true_density(prediction, target):
    return {
        'metric': torch.mean(target['charge']).item(),
        'total': torch.sum(target['charge']).item(),
        'numel': target['charge'].numel(),
    }

def norm_charge_mae(prediction, target):
    error = torch.sum(torch.abs(prediction['charge'] - target['charge'])) \
          / torch.sum(torch.abs(target['charge']))
    return {
        'metric': error.item(),
        'total':  error.item(),
        'numel':  1,
    }
        
def norm_charge_rmse(prediction, target):
    error = torch.sqrt(torch.sum(torch.square(prediction['charge'] - target['charge'])) \
          / torch.sum(torch.abs(target['charge'])))
    return {
        'metric': error.item(),
        'total':  error.item(),
        'numel':  1,
    }
