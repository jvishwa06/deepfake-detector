# Deepfake Detector

A web app that checks whether a face in an image is real or a deepfake, and shows which parts of the face influenced the decision.

## How it works

`main.py` starts a [Gradio](https://www.gradio.app) interface. When you upload an image it:

1. Finds the face with MTCNN (`facenet-pytorch`). It reports an error if no face is found.
2. Resizes the face to 256x256 and classifies it with an InceptionResNetV1 model (VGGFace2 weights, one output) loaded from `models/inceptionresnetv1.pth`.
3. Shows the confidence for `real` and `fake`.
4. Draws a Grad-CAM heat map over the face, so you can see which regions the model focused on.

## Training

`notebooks/training.ipynb` trains the model:

- **Data**: faces extracted from FaceForensics++ videos (original actor sequences and DeepFakeDetection manipulated sequences), listed in a `faces.csv` file with a 20% validation split. The dataset itself is not in this repo.
- **Setup**: InceptionResNetV1, `BCEWithLogitsLoss`, Adam optimiser (learning rate 0.001), batch size 32, 80 epochs, with a checkpoint every 4 epochs.
- **Tracking**: metrics are logged with Weights & Biases.
- Helper code is in `notebooks/dataset/` and `notebooks/utils/`.

## Run

Requires Python 3.11 or newer. The dependencies are in `pyproject.toml` (PyTorch, `facenet-pytorch`, `grad-cam`, Gradio, OpenCV, Albumentations and others).

```bash
uv sync
uv run python main.py
```
