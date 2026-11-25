import numpy as np
import torch

from frugaloptix.ml import Classifier, Restorer


def test_restorer_starts_as_identity_and_keeps_shape():
    net = Restorer(8).eval()
    x = torch.rand(2, 3, 64, 64)
    assert net(x).shape == x.shape
    assert torch.allclose(net(x), x, atol=1e-6)


def test_classifier_outputs_one_logit_per_image():
    out = Classifier(False).eval()(torch.rand(3, 3, 64, 64))
    assert out.shape == (3,) and np.isfinite(out.detach().numpy()).all()
