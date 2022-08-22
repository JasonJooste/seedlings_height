# Adding height data to improve the detection of seedlings using Faster-RCNN

Addition of height map data to drone images for the detection of seedlings.
Five different model structures are tested:

- The unmodified Faster-RCN network (Vanilla)
- Height as a fourth image channel to the first convolutional layer (First)
- Height as an input after region of interest pooling and before the final dense classification and bounding box
  adjustment heads (Final)
- Height concatenated to the output of the backbone before RoI pooling, but after the region proposal network (pre-RoI)
- Height concatenated to the output of the backbone before RoI pooling and the RPN (Pre-RPN)

These models are first generated (see src/models) and saved (in models/templates) and then loaded, trained, tested and
stored for a given hyperparameter selection.
The results are the saved to an MLFlow server.

# Usage

Run the main file as a package with configuration files as arguments. The cartesian product of all iterables in the
config file are then used to train and test a model, with the results saved in MLFlow.
See the existing configuration files for an example of how to format them.
For example:

```shell
python3 -m src/main models/configs/21_06_17_first_test_long_cv2.yaml
```

<!-- unfortunately the location of the MLFlow server needs to be manually edited in the main and analysis-functions files>
