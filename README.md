# Yolov3-Pytorch
Pytorch implementation of [Yolov3](https://arxiv.org/abs/1804.02767v1)

## Requirements
- Python 3.5 
- Pytorch 4.1

I will not be considering about compatible issues, such as division signs difference between 
Python 2.7 and Python 3+, So please do use Python 3.5+

I tried to use Pytorch 3.1 as my main choice, but it was just not that greate, for example, it
didnt support index slicing quite well,
when I type this in Pytorch 3.1:
```Python
objectness_mask[ious > self.ignore_thresh, :] = 0
```
it gives me this error:
```
TypeError: Performing basic indexing on a tensor and encountered an error indexing dim 1 with an object of type torch.cuda.LongTensor. The only supported types are integers, slices, numpy scalars, or if indexing with a torch.LongTensor or torch.ByteTensor only a single Tensor may be passed.
```
But when I run the same code in Pytorch 4.1, everything works fine, it seems Pytorch 3.1 doesnt have full support for advanced
indexing, and full support was added in Pytorch 4.0

