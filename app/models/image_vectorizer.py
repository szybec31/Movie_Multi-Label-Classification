# app/models/image_vectorizer.py

import os
import numpy as np
import torch

from PIL import Image

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


class ImageVectorizer:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        weights = ResNet50_Weights.DEFAULT

        model = resnet50(weights=weights)

        # remove classifier
        self.model = torch.nn.Sequential(
            *list(model.children())[:-1]
        )

        self.model.eval()
        self.model.to(self.device)

        self.transform = weights.transforms()

    # ========================
    # SINGLE IMAGE
    # ========================
    def encode(self, image_path):

        try:

            image = (
                Image.open(image_path)
                .convert("RGB")
            )

            image = self.transform(image)

            image = image.unsqueeze(0)

            image = image.to(self.device)

            with torch.no_grad():

                features = self.model(image)

            features = (
                features
                .cpu()
                .numpy()
                .flatten()
            )

        except Exception as e:

            print("Image encoding error:", e)

            features = np.zeros(2048)

        return features