# Technion Multi-Node Tutorial

At the moment, this tutorial uses [Horovod](https://horovod.ai/) for an easy and quick start. For advanced configuration please consider using other methods (such as [DDP](https://pytorch.org/docs/stable/notes/ddp.html) for PyTorch).

## Introduction

The following applications and tools are involved in the process. Each is quickly explained in relation to this tutorial.

- [NVIDIA NGC](https://catalog.ngc.nvidia.com/) is a catalog for resources, including accelerated containers of famous frameworks such as PyTorch and TensorFlow.
  - __Note:__ It is recommended to use these containers as a base "engine" for your applications.
- [NVIDIA Enroot](https://github.com/NVIDIA/enroot) is used to pull and modify containers.
- [NVIDIA Pyxis](https://github.com/NVIDIA/pyxis) is a Slurm's plugin for running containers.
- [Slurm](https://slurm.schedmd.com/documentation.html) is a workload manager for allocating resources (nodes, tasks, GPUs, etc.) needed for a job (a container) to run, and provides the MPI library.
- [MPI](https://www.open-mpi.org/) is used for providing communication between resources.
- Horovod uses MPI and the resources to run the relevant application.
  - __Note:__ without Slurm, Horovod should be used with ["horovodrun"](https://horovod.readthedocs.io/en/stable/running_include.html) for multi-node and multi-GPUs applications.

## Preparing a container

Pulling, modifying and preparing a container for running was previously explained in the [Technion's Pre-Slurm Tutorial](https://gitlab.com/anahum/technion-users-pre-slurm-tutorial).

- __Note:__ [NVIDIA NGC TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) containers are already equipped with Horovod. Other containers and frameworks that lack of it should be modified and Horovod should be installed prior to running the application. To install Horovod please follow [Horovod Installation Guide](https://horovod.readthedocs.io/en/stable/install_include.html). For example, to install Horovod in a PyTorch container:
  - Run the container with Enroot.
  - Install Horovod with `pip install horovod[pytorch]`.

## Preparing your code

Implementing Horovod in your code is fairly a simple process and is well [documented](https://github.com/horovod/horovod#usage) with some examples. Another very basic PyTorch neural network example with a single forward and backward passes is presented here:

```python
import torch
import torch.nn as nn
import torch.optim as optim

import horovod.torch as hvd


def example():
    print("start horovod")
    hvd.init()

    print("-" * 20)
    print("total number of GPUs:", hvd.size())
    print("number of GPUs on this node:", hvd.local_size())
    print("rank:", hvd.rank())
    print("local rank:", hvd.local_rank())
    print("-" * 20)

    print("create a local model")
    torch.cuda.set_device(hvd.local_rank())
    model = nn.Linear(10, 10)
    model.cuda()

    print("define a loss function")
    loss_fn = nn.MSELoss()
    print("define an optimizer")
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    print("distribute parameters")
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    print("run a forward pass")
    outputs = model(torch.randn(20, 10).cuda()).cuda()
    labels = torch.randn(20, 10).cuda()

    print("run a backward pass")
    loss_fn(outputs, labels).backward()
    print("update parameters using the optimizer")
    optimizer.step()

    print("done")

def main():
    example()

if __name__=="__main__":
    main()

```

## Submitting a Slurm job

Slurm jobs should be submitted via [SBATCH](https://slurm.schedmd.com/sbatch.html) script files. A basic example of such file is presented here:

```bash
#!/bin/bash
#SBATCH --job-name=<job name>
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --ntasks=<number of total GPUs>
#SBATCH --gpus=<number of total GPUs>

srun --container-image=<path to container image> \
     --container-mounts=<path to code>:/code \
     --no-container-entrypoint \
     /bin/bash -c \
     "python -u <path to code Python's file>"

```

__Note:__

- `SBATCH` lines provides the resources. `srun` provides the command.
- Two files will be created named with `slurm-<job name>-<job id>`. The `.out` file provides the regular output, while the `.err` file provides the errors.
- In case specific type of GPUs should be used, use `--gpus=<GPU type>:<number of GPUs>`. E.g., `--gpus=a100:2` for 2 A100 GPUs.
- Horovod needs enough tasks to use all of the GPUs. Therefore, the number of tasks provided in `--ntasks` should be equal to the number of GPUs. A lack of tasks will result in GPUs being allocated but not used.
- To output each task to a different file add `--output=slurm-%x-%j-%t.out` to the `srun` command. This will create a new file for each task named `slurm-<job name>-<job id>-<task id>.out`. Usually, the main output will be available in `slurm-<job name>-<job id>-0.out` file.
- More information is available in Slurm's [srun](https://slurm.schedmd.com/srun.html) and [sbatch](https://slurm.schedmd.com/sbatch.html) documentations.

## Example

Horovod provides [examples](https://github.com/horovod/horovod/tree/master/examples) for running applications. To see a quick multi-node example in action use the following:

- Pull an NVIDIA NGC [TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) container:

  ```bash
  enroot import 'docker://nvcr.io#nvidia/tensorflow:22.09-tf1-py3'
  ```

- Clone Horovod's GitHub repository:

  ```bash
  git clone https://github.com/horovod/horovod
  ```

- Create a new SBATCH script file and name it `example.sub`:

  ```bash
  #!/bin/bash
  #SBATCH --job-name=horovod_example
  #SBATCH --output=horovod_example.out
  #SBATCH --error=horovod_example.err
  #SBATCH --ntasks=16
  #SBATCH --gpus=16

  srun --container-image=<path to TensorFlow container image> \
       --container-mounts=<path to Horovod GitHub folder>:/code \
       --no-container-entrypoint \
       /bin/bash -c \
       "python -u /code/examples/tensorflow/tensorflow_synthetic_benchmark.py"
  ```

  - __Note:__ this example uses 16 GPUs which guarantees the use of at least two DGX A100 nodes. You can decrease the number of GPUs, but to guarantee the use of more than a single node please add `#SBATCH --nodes=<number of nodes>`. A use of 3 GPUs and 2 nodes is recommended to observe a minimal imbalanced resources example of a multi-node run.

- Submit the job:

  ```bash
  sbatch <path to example.sub>
  ```

- Examine the output file:

  ```bash
  vi horovod_example.out
  ```

__Note:__ to view which resources were allocated to the job run the following command:

```bash
scontrol show -d job <job id> | grep GRES
```
