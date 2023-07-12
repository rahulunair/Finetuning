# FineTuner

Finetuning code to build a classifier for forest fires. Works on both intel CPUs and XPUs.

To set the device, edit the set_device call in config.py


Batch size, learning rate finder and data augmetation can be turned off or on in finetune.py using flags.

```bash
python fine_tune.py
```

Output:

```bash
XPU devices available, using xpu:8
XPU device name: Intel(R) Data Center GPU Max 1550
OMP_NUM_THREADS set to: 112
Augmenting dataset...
..
..
```
