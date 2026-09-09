# The plant-disease model

## Status: working

The classifier is installed and serving real predictions. `/api/health` reports
`models.disease.ready: true`.

Verified against the original project's own published screenshot
(`Plant-Disease-Prediction-main/Images/image2.png`), which shows a specific
grape leaf detected as **`Grape___Black_rot` at 100.00%**. This service returns
the identical label and confidence for that image, which confirms both the
weights and the preprocessing match the original.

## Where the weights came from

The `.h5` file is not in the source repository and never was. The original
README says to download it separately:

> Please download the `plant_disease_prediction.h5` file through this drive link

That Google Drive link now returns **404**, and the upstream GitHub repo
(`21lakshh/Kisaan-Saathi`) contains no model file and has no releases.

The surviving copy was inside the author's published Docker image. It was
recovered from the registry without pulling the whole 1.9 GB image:

```bash
# 1. anonymous pull token
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:21laksh/kisaan-saathi-image:pull" \
  | python -c "import json,sys;print(json.load(sys.stdin)['token'])")

# 2. the manifest names the layers; the ~237 MB one is the `COPY . /app` step
curl -s -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://registry-1.docker.io/v2/21laksh/kisaan-saathi-image/manifests/v1.0"

# 3. fetch that one blob (supports byte-range resume; tokens expire in minutes)
curl -L -C - -H "Authorization: Bearer $TOKEN" \
  "https://registry-1.docker.io/v2/21laksh/kisaan-saathi-image/blobs/sha256:aa2058d1073a64790d10ae847915564e0d8c8fdcb603ed436e7411f0b15b3373" \
  -o layer.tar.gz

# 4. verify, then extract only the model
sha256sum layer.tar.gz   # must equal the digest above
tar -xzf layer.tar.gz app/trained_model/plant_disease_prediction.h5
```

The layer also contains the author's `config.json`, which holds their Google API
key. Extract only the model file; do not use that key.

## What the model is

From the training notebook (`ml-service/training/train_disease.ipynb`):

| | |
|---|---|
| Architecture | Keras `Xception`, ImageNet weights, `include_top=False`, `pooling='avg'` |
| Head | `BatchNormalization` → `Dense(256, relu)` → `Dropout(0.5)` → `Dense(38, softmax)` |
| Input | 224 × 224 × 3 |
| Preprocessing | **rescale by `1/255`** — the notebook used `ImageDataGenerator(rescale=1./255)`, *not* Xception's `preprocess_input` |
| Classes | 38 PlantVillage labels, listed in `artifacts/disease/class_indices.json` |
| Dataset | [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| Reported accuracy | 99.72 % on the held-out test set after fine-tuning |
| File | 245 MB, HDF5, written by Keras 3.5.0 |

The `1/255` detail matters. Using `preprocess_input` instead would scale inputs
to [-1, 1] and quietly degrade accuracy without raising an error, so
`app/routers/disease.py` pins the rescale and says why.

## Why loading needs a rebuild

The distributed `.h5` was written by **Keras 3.5.0** and cannot be deserialised
by either current major version:

- **Keras 3.15** fails on the nested Xception's build config:
  `ValueError: Cannot convert '((None, 2048),)' to a shape`
- **Keras 2** (`tf_keras`) fails because Keras 3 wrote `batch_shape` where
  Keras 2 expects `batch_input_shape`:
  `Unrecognized keyword arguments: ['batch_shape']`

`app/disease_model.py` sidesteps the serialised config entirely: it rebuilds the
architecture from the notebook — with the layer names pinned to the ones in the
file — and loads only the weight arrays. That is version-independent.

It also guards against silent failure: if no weights match, the output layer
would still be randomly initialised and every prediction would be noise, so the
loader compares the layer before and after and raises instead.

On first load the model is re-saved as `plant_disease_prediction.keras` (native
Keras 3), and `config.py` prefers that file, so subsequent starts skip the
rebuild. Startup is about 8 s for this model, once per process.

## Installing it elsewhere

Drop either file into `ml-service/artifacts/disease/`:

- `plant_disease_prediction.keras` — preferred, loads directly
- `plant_disease_prediction.h5` — the original, goes through the rebuild path

Or point `DISEASE_MODEL_PATH` at it anywhere on disk.

TensorFlow is not in the base requirements — it is ~600 MB and only this
endpoint needs it:

```bash
cd ml-service
python -m pip install -r requirements-disease.txt
```

Then restart and check `models.disease.ready` at
<http://localhost:8100/health>. If it is `false`, the `error` field says why.

Version note: TensorFlow 2.20 (Keras 3.15) on Python 3.13 is what this was
verified against. The model was trained on 2.18, which has no 3.13 wheel; the
rebuild path makes the version difference irrelevant.

## Retraining

`ml-service/training/train_disease.ipynb` is the original notebook. It needs the
PlantVillage dataset (~2 GB, 38 class folders) and a GPU; the fine-tuning run
takes roughly an hour on Colab.

Two things to keep unchanged so the service can load the output:

1. Save as `.h5` or `.keras` with the same 38-class output order.
2. If the class order changes, regenerate `class_indices.json` — the notebook's
   `json.dump(train_generator.class_indices, ...)` cell does this — and copy it
   to `ml-service/artifacts/disease/`.

The registry reads that file as `{"0": "Apple___Apple_scab", ...}`, index to
label. A mismatch between it and the model's output layer would map every
prediction to the wrong disease, so regenerate them together.

If you change the architecture, update `build_architecture()` in
`app/disease_model.py` to match, or save in `.keras` format so no rebuild is
needed.
