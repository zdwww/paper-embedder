"""Provider protocol: any class satisfying the protocol can be used as a Provider."""

import numpy as np


def test_provider_protocol_accepts_conforming_class():
    from paper_embedder.providers.base import Provider

    class MyProvider:
        name = "fake"
        dim = 8
        max_input_tokens = 100

        def embed(self, texts, *, mode):
            return [np.zeros(self.dim, dtype=np.float32) for _ in texts]

        def truncate(self, text):
            return text

        def fingerprint(self):
            return "abc"

    p: Provider = MyProvider()
    assert p.name == "fake"
    assert p.dim == 8
    assert len(p.embed(["hi"], mode="document")) == 1
    assert p.truncate("x" * 1000) == "x" * 1000
    assert p.fingerprint() == "abc"
