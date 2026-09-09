# Test fixtures

**`grape_black_rot.png`** — a grape leaf with black-rot lesions, cropped from
`Plant-Disease-Prediction-main/Images/image2.png`, a screenshot the original
project published showing its own app detecting this exact leaf as
`Grape___Black_rot` at 100.00% confidence.

That published result is the ground truth. `test_api.py` asserts our pipeline
returns the same label, which verifies three things at once: the weights are the
author's trained model, `class_indices.json` maps indices to labels correctly,
and the preprocessing (224x224, rescale 1/255) matches how the model was
trained. A synthetic image could not check any of those.
