# FineTuner

FineTuner is a robust framework for building a forest fire classifier. It provides support for both Intel CPUs and XPUs, and includes several customizable options to fine-tune model performance.

## Features
- Device selection: You can select your processing unit (CPU or XPU) in the configuration file. If not explicitly set, the default device is selected based on availability.
- Optimal settings finder: The framework includes flags for finding the optimal batch size and learning rate.
- Data augmentation: You can choose to augment your data set to create a larger training corpus if the original dataset is small.

## Installation
Run the `setup.sh` bash script to install all the necessary dependencies, including PyTorch, IPEX, and Torchvision.

```bash
./setup.sh
```

## Usage

Run the `fine_tune.py` script to start the model fine-tuning process. The main function in this script allows you to configure various parameters such as data augmentation, learning rate finder, batch size finder, WandB, and IPEX usage. By default, all options are set to False except for IPEX.

```bash
python fine_tune.py
```

## Sample Output

Here's a sample output of the script:

```bash
OMP_NUM_THREADS set to: 112
XPU devices available, using xpu:8
XPU device name: Intel(R) Data Center GPU Max 1550

Running in Finetuning mode.
...

Model saved to :./models/model_acc_45_device_xpu_lr_0.00214_epochs_1.pt
Time elapsed: 151.98359322547913 seconds.
```



