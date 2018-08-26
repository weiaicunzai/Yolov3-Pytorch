# Yolov3-Pytorch (under construction.........)
Pytorch implementation of [Yolov3](https://arxiv.org/abs/1804.02767v1)

## Requirements
- Python 3.5 
- Pytorch 4.1

I will not be considering about compatible issues, such as division signs difference between 
Python 2.7 and Python 3+, So please do use Python 3.5+.Also, using Pytorch3.1 or below will cause 
program to collape

I tried to use Pytorch 3.1 as my main choice, but it was just not that greate, for example, it
didnt fully support index slicing.
when I type this in Pytorch 3.1:
```Python
objectness_mask[ious > self.ignore_thresh, :] = 0
```
it gives me this error:
```
TypeError: Performing basic indexing on a tensor and encountered an error indexing dim 1 with an object of type torch.cuda.LongTensor. The only supported types are integers, slices, numpy scalars, or if indexing with a torch.LongTensor or torch.ByteTensor only a single Tensor may be passed.
```
But when I run the same code in Pytorch 4.1, everything works fine, it seems Pytorch 3.1 doesnt have full support for advanced
indexing, and the full support was not added untill Pytorch 4.0, so I decided to upgrade my Pytorch to 4.1, cause using for
loop to assign and compare value is just a pain in the ass.

