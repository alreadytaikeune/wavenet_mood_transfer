#! encoding=utf8
from __future__ import unicode_literals, print_function, absolute_import

from wavenet import WaveNet


class MoodTransferModel(object):

    def __init__(self, batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32):
        self.encoder = WaveNet()
        self.decoder = WaveNet()

        self.classifier = None
        self.built = False

    def downsample(self, x):
        pass

    def upsample(self, x):
        pass

    def build(self, waveform, mood_repr):
        self.inputs = [waveform, mood_repr]
        X_e = self.encoder.predict_proba(waveform, name="encoder")
        X_l = self.downsample(X_e)
        X_u = self.upsample(X_l)
        X_out = self.decoder.predict_proba_in(
            X_u, name="decoder")
        self.X_e = X_e
        self.X_l = X_l
        self.X_u = X_u
        self.X_out = X_out
        self.classifier = self._create_classifier(X_l)
        self.built = True

    def classification_loss(self):
        if not self.built:
            raise ValueError("")
        pass

    def reconstruction_loss(self):
        pass

    def __call__(self, waveform, mood_repr):
        self.build(waveform, mood_repr)
        return self.X_out
