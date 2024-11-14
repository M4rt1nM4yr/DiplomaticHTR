# Data-Efficient Handwritten Text Recognition of Diplomatic Historical Text

This repository contains the PyTorch implementation of a model designed for data-efficient handwritten text recognition, as outlined in the paper: "Data-Efficient Handwritten Text Recognition of Diplomatic Historical Text".

## Building the Model

You can construct the model with the following code snippet:

```python
from functools import partial
from src.cnn.resnet import Resnet
from src.encoder.htr_transformer_encoder import HTRTransformerEncoder
from src.decoder.htr_transformer_decoder import HTRTransformerDecoder
from src.utils.alphabet import Alphabet

rezero = True
alphabet = Alphabet('ab')
cnn = Resnet(in_channels=1, wider=True)
encoder = partial(
    HTRTransformerEncoder,
    d_model=256,
    rezero=rezero,
)
decoder = partial(
    HTRTransformerDecoder,
    d_model=256,
    rezero=rezero,
)
m = S2SDipl(
    alphabet=Alphabet('ab'),
    cnn=cnn,
    encoder=encoder,
    decoder=decoder,
).cuda()
```

## Usage Example

For a complete example of how to use this model, refer to src/model.py, which includes a sample implementation.

## Requirements

Ensure you have the necessary dependencies installed. See requirements.txt
