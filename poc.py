import madmom


class SpectralFlux:
    def __init__(self, wav_file):
        """
        Creates spectrogram, and then runs spectral flux for onset detection
        """
        spec = madmom.audio.spectrogram.Spectrogram(
            wav_file, frame_size=2048, hop_size=200, fft_size=4096
        )
        return madmom.features.onsets.spectral_flux(spec)


def main():
    pass  # func would probably loop through folder with training data and init spectral flux


if __name__ == "__main__":
    main()
