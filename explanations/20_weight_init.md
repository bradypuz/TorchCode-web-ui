# Kaiming Initialization

## Goal and intuition

Choose the initial size of a linear layer's weights so signals do not immediately shrink or explode as they move through ReLU layers. More incoming features mean more terms are added at each neuron, so each weight should generally be smaller. Kaiming normal initialization draws zero-centered weights with variance `2 / fan_in`.

## Inputs and outputs

- `weight`: an existing floating tensor, normally `(out_features, in_features)` in this exercise.
- Return value: the **same tensor object**, filled in place with new random values.
- Its shape, device, and dtype are retained.

For a 2-D linear weight matrix, `fan_in` is the number of input features, `weight.shape[1]`. The reference also accepts a 1-D tensor and uses its length as `fan_in`.

## Walk through the reference

1. Compute `fan_in` from the second dimension when the tensor has at least two dimensions, otherwise from the first dimension.
2. Calculate `std = math.sqrt(2.0 / fan_in)`.
3. Enter `torch.no_grad()`. Initialization mutates parameter values and should not become an operation in the training computation graph.
4. Call `weight.normal_(0, std)` to overwrite the tensor with samples from the chosen normal distribution.
5. Return `weight` rather than allocating a replacement tensor.

## Why it works

$$
W_{ij}\sim\mathcal{N}\left(0,\frac{2}{\mathrm{fan\_in}}\right)
$$

Under the usual independence and roughly symmetric-activation assumptions, a linear sum's variance grows with `fan_in`. ReLU removes roughly half of a symmetric signal's second moment. The factor of two compensates for that reduction. This is a statistical initialization rule, not a guarantee that every layer or finite random sample has exactly the target variance.

## Pitfalls and reference scope

`normal_` takes a **standard deviation**, so use `sqrt(2/fan_in)`, not `2/fan_in`. Finite samples have an approximately zero mean and approximately the expected standard deviation. The reference is tailored to linear weights: for convolution weights it **ignores kernel height and width**, which belong in a full convolutional fan-in calculation. It has no fan-out mode, configurable nonlinearity gain, or uniform variant. A scalar tensor has no first dimension, and a zero fan-in causes division by zero; neither case is validated. The judge checks large 2-D matrices, in-place identity, and the effect of different fan-in values.
