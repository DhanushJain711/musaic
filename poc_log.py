import madmom
import numpy as np


class SuperFlux: #spectral flux but with log filtering instead of linear filtering
    def __init__(self, num_bands=24, diff_max_bins=3, positive_diffs=True):
        """
        This init allows us to set optional parameters before doing log filtering
        """
        self.num_bands = num_bands
        self.diff_max_bins = diff_max_bins
        self.positive_diffs = positive_diffs
        

    def process(self, data):
        """
        runs actual superflux, which is log filtered instead of linearly filtered
        """
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
