# Train and test on CVPPP dataset
In this example we use zero_padding convolutional UNet, that means that the coloring procedure will use the knowledge about borders to do coloring.


## Run training

```shell
python train_cvppp.py path_to_A1_dataset
python3 train_cvppp.py -s /Users/lisulimowicz/Downloads/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Stacks/Ara2012/stack_04
```

## Notes on code
Creates a batch generator

```python
generator = train_data.create_batch_generator(30, transforms=transforms)
```
---
Creates a halo region function

```python
mask_builder = dc.build_halo_mask(fixed_depth=30, margin=21, min_fragment=10)
```

1. fixed_depth - maximum number of object in a training batch
2. margin - size of margin (dilatation) around the object sould be odd
3. min_fragment - minimal size of an object in pixels
---
Training
```python
model, errors = dc.train(generator=generator,
                             model=net,
                             mask_builder=mask_builder,
                             niter=10000,
                             k_neg=7.,
                             lr=1e-3,                             
                             caption=join(directory, "model"))
```
1. generator - batch generator
2. model - segmentation network
3. niter - number of iterations
4. k_neg - balance between positive and negative parts of loss please seen paper
5. lr - learining rate
6. caption - name of errors file and model
