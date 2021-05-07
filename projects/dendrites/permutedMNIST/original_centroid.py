# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
The original implementation of the centroid method for continual learning with
dendritic networks, which achieves ~70% accuracy on 50 tasks.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins
from nupic.torch.modules import KWinners, SparseWeights, rezero_weights
from projects.dendrites.gaussian_classification.gaussian import GaussianDataset


# ------ Experiment class
class CentroidContextExperiment(mixins.RezeroWeights,
                                ContinualLearningExperiment):

    def setup_experiment(self, config):
        self.batch_size = config.get("batch_size", 1)
        self.val_batch_size = config.get("val_batch_size", 1)

        super().setup_experiment(config)

        self.contexts = torch.zeros((0, self.model.input_size))
        self.contexts = self.contexts.to(self.device)


# ------ Network
class DendriticMLP(nn.Module):
    """
    - different version of `DendriticMLP` than current version in master
    - context is either provided as a dense vector (to which kW is applied), or the
    input is used (again, to which kW is applied) """
    def __init__(self, input_size, output_size, hidden_size, num_segments, dim_context,
                 dendrite_sparsity, kw,
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):

        super().__init__()

        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dim_context = dim_context
        self.kw = kw

        # Forward layers & k-winners
        self.dend1 = dendritic_layer_class(
            module=nn.Linear(input_size, hidden_size, bias=True),
            num_segments=num_segments,
            dim_context=dim_context,
            module_sparsity=0.95,
            dendrite_sparsity=dendrite_sparsity
        )
        self.dend2 = dendritic_layer_class(
            module=nn.Linear(hidden_size, hidden_size, bias=True),
            num_segments=num_segments,
            dim_context=dim_context,
            module_sparsity=0.95,
            dendrite_sparsity=dendrite_sparsity
        )

        if kw:
            print("Using k-Winners, 0.05 'on'")
            self.kw1 = KWinners(n=hidden_size, percent_on=0.05, k_inference_factor=1.0,
                                boost_strength=0.0, boost_strength_factor=0.0)
            self.kw2 = KWinners(n=hidden_size, percent_on=0.05, k_inference_factor=1.0,
                                boost_strength=0.0, boost_strength_factor=0.0)

        self.classifier = SparseWeights(module=nn.Linear(hidden_size, output_size),
                                        sparsity=0.95)

        if kw:
            self.kw_context = KWinners(n=dim_context, percent_on=0.05,
                                       k_inference_factor=1.0, boost_strength=0.0,
                                       boost_strength_factor=1.0)

        # Scale weights to be sampled from the new inititialization U(-h, h) where
        # h = sqrt(1 / (weight_density * previous_layer_percent_on))
        init_sparse_weights(self.dend1, 0.0)
        init_sparse_weights(self.dend2, 0.95 if kw else 0.0)
        init_sparse_weights(self.classifier, 0.95 if kw else 0.0)

        # Do the same for dendrites
        init_sparse_dendrites(self.dend1, 0.95)
        init_sparse_dendrites(self.dend2, 0.95)

    def forward(self, x, context=None):
        # Context processing
        if context is None:
            context = x

        context = self.kw_context(context)

        # Forward processing
        output = self.dend1(x, context=context)
        output = self.kw1(output) if self.kw else output

        output = self.dend2(output, context=context)
        output = self.kw2(output) if self.kw else output

        output = self.classifier(output)
        return output


# ------ Weight initialization functions
def init_sparse_weights(m, input_sparsity):
    input_density = 1.0 - input_sparsity
    if hasattr(m, "sparsity"):
        weight_density = 1.0 - m.sparsity
        module = m.module
    else:
        # `m` isn't a sparse weight module; weights are dense
        weight_density = 1.0
        module = m
    _, fan_in = module.weight.size()
    bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
    nn.init.uniform_(module.weight, -bound, bound)
    m.apply(rezero_weights)


def init_sparse_dendrites(m, input_sparsity):
    """ Assume `m` is an instance of `DendriticLayerBase` """
    input_density = 1.0 - input_sparsity
    weight_density = 1.0 - m.segments.sparsity
    fan_in = m.segments.dim_context
    bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
    nn.init.uniform_(m.segments.weights, -bound, bound)
    m.apply(rezero_weights)


# ------ Training & evaluation function
def train_model(exp, context):
    exp.model.train()
    context = context.to(exp.device)
    context = context.repeat(exp.batch_size, 1)
    for batch_item in exp.train_loader:
        data = batch_item[0]
        data = data.flatten(start_dim=1)
        target = batch_item[-1]

        if exp.model.input_size == 784:
            # permutedMNIST requires a single head
            target = target % exp.num_classes_per_task

        data, target = data.to(exp.device), target.to(exp.device)

        exp.optimizer.zero_grad()
        output = exp.model(data, context)

        output = F.log_softmax(output)
        error_loss = exp.error_loss(output, target)

        error_loss.backward()
        exp.optimizer.step()

        # Rezero weights if necessary
        exp.post_optimizer_step(exp.model)


def evaluate_model(exp):
    exp.model.eval()
    total = 0

    loss = torch.tensor(0., device=exp.device)
    correct = torch.tensor(0, device=exp.device)

    with torch.no_grad():

        for batch_item in exp.val_loader:
            data = batch_item[0]
            data = data.flatten(start_dim=1)
            target = batch_item[-1]

            if exp.model.input_size == 784:
                # permutedMNIST requires a single head
                target = target % exp.num_classes_per_task

            data, target = data.to(exp.device), target.to(exp.device)

            # Select the context by comparing distances to all context prototypes
            context = torch.cdist(exp.contexts, data)
            context = context.argmin(dim=0)
            context = context.cpu()
            context = exp.contexts[context]

            output = exp.model(data, context)

            # All output units are used to compute loss / accuracy
            loss += exp.error_loss(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    mean_acc = torch.true_divide(correct, total).item() if total > 0 else 0,
    return mean_acc


def centroid(exp):
    # Computes the centroid vector over all examples in loader
    c = torch.zeros((exp.model.input_size,))
    c_size = 0
    for batch_item in exp.train_loader:
        x = batch_item[0]
        x = x.flatten(start_dim=1)

        x_size = x.size(0)
        c_size = c.size(0)

        c = c.to(x.device)

        c = c_size * c + x.sum(dim=0)
        c = c / (c_size + x_size)
        c_size = c_size + x_size

    return c


if __name__ == "__main__":

    num_tasks = 2
    num_epochs = 1

    dataset = "permutedmnist"

    if dataset == "permutedmnist":
        config = dict(
            dataset_class=PermutedMNIST,
            dataset_args=dict(
                num_tasks=num_tasks,
                seed=np.random.randint(0, 1000)
            ),

            model_args=dict(
                input_size=784,
                output_size=10,
                hidden_size=2048,
                num_segments=num_tasks,
                dim_context=784,
                dendrite_sparsity=0.0,
                kw=True,
            ),

            batch_size=256,
            epochs=1,
            num_classes=10 * num_tasks,
        )

    elif dataset == "gaussian":
        config = dict(
            dataset_class=GaussianDataset,
            dataset_args=dict(
                num_classes=2 * num_tasks,
                num_tasks=num_tasks,
                training_examples_per_class=2500,
                validation_examples_per_class=125,
                dim_x=2048,
                dim_context=2048,
            ),

            model_args=dict(
                input_size=2048,
                output_size=2 * num_tasks,
                hidden_size=2048,
                num_segments=num_tasks,
                dim_context=2048,
                dendrite_sparsity=0.0,
                kw=True,
            ),

            batch_size=64,
            epochs=1,
            num_classes=2 * num_tasks,
        )

    config.update(
        experiment_class=CentroidContextExperiment,

        val_batch_size=512,
        epochs_to_validate=np.arange(1),
        num_tasks=num_tasks,

        model_class=DendriticMLP,
        distributed=False,
        seed=np.random.randint(0, 10000),
        loss_function=F.nll_loss,
        optimizer_class=torch.optim.Adam,
        optimizer_args=dict(lr=1e-3),
    )

    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)

    ############################# CONTINUAL LEARNING PHASE ############################

    optimizer_class = config.get("optimizer_class", torch.optim.SGD)
    optimizer_args = config.get("optimizer_args", {})

    for task_id in range(num_tasks):

        # Train model on current task
        exp.train_loader.sampler.set_active_tasks(task_id)

        # Build context vectors through centroid
        context = centroid(exp).to(exp.device).detach()
        exp.contexts = torch.cat((exp.contexts, context.unsqueeze(0)))

        for _ in range(num_epochs):
            train_model(exp, context)

        if task_id % 5 == 4:
            print(f"=== AFTER TASK {task_id} ===")
            print("")

            # Evaluate model accuracy on each task separately
            for eval_task_id in range(task_id + 1):

                exp.val_loader.sampler.set_active_tasks(eval_task_id)
                acc_task = evaluate_model(exp)
                if isinstance(acc_task, tuple):
                    acc_task = acc_task[0]

                print(f"task {eval_task_id} accuracy: {acc_task}")

            print("")

        # Reset optimizer before starting new task
        del exp.optimizer
        exp.optimizer = optimizer_class(exp.model.parameters(), **optimizer_args)

    ###################################################################################

    # Final aggregate accuracy
    exp.val_loader.sampler.set_active_tasks(range(num_tasks))
    acc_task = evaluate_model(exp)
    if isinstance(acc_task, tuple):
        acc_task = acc_task[0]

    print(f"Final test accuracy: {acc_task}")
