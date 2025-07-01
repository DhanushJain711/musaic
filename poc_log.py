import madmom
import numpy as np


class SuperFluxWithLogFiltering:
    def __init__(self, num_bands=24, diff_max_bins=3, positive_diffs=True):
        self.num_bands = num_bands
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs

    def process(self, data):
        spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(
            data, num_bands=self.num_bands
        )
        diff = madmom.audio.spectrogram.SpectrogramDifference(
            spec, diff_max_bins=self.diff_max_bins, positive_diffs=self.positive_diffs
        )
        return np.mean(diff, axis=1)


def main():
    pass  # func would probably loop through folder with training data and init spectral flux


if __name__ == "__main__":
    main()
