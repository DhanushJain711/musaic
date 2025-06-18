# Musaic

## AI Pipeline
The basic pipeline to go from a recording of you playing your instrument to getting understandable feedback on what to improve:

1. Take a short recording of instrumental music (i.e. someone practicing) and split the recording into individual notes.

    ### Considerations:
    - Lots of options here. There are multiple python packages which can perform note segmentation, most are based around generating graphs through FFT and detecting note attacks and decays.
    - Librosa, Essentia (audio engineering based), Madmom (deep learning based)
    - We wil need to decide what the max length of the recordings we process. We don't want to take in too long of a recording and have to process too many notes. One example is the music is fast or very technical (high density of notes per second), then the pipeline might have too much to handle.

2. Use deep learning to classify the individual notes according to various measures such as intonation and articulation. These models will be trained off of the [GoodSounds](https://paperswithcode.com/dataset/goodsounds) dataset.

    ### Considerations:
    - Will probably store at least some part of the dataset on Drive or something to be collaboratively accessed, the dataset is around 15 gb.
    - Which models do we want to use?
        - Classical CNNs on audio spectograms, generated through FFT
        - More advanced models such as vision transformers (DETR) on spectorgrams
        - Wav2Vec: converting audio to vectors and working from there, works well for voice but not sure about musical notes
        - Sequence to sequence models that can classify multiple notes at at time and could possibly include attention mechanisms? (LSTM, encoder-decoder)
        - From the paper: extract specific, well defined features from the recordings and train classic ML models on them (boring...)

3. Once we have the individual classifications for the notes, we can feed this data in a structured way to an LLM. We can request that LLM to summarize the classifications to determine general trends, areas of improvement, and present this in an understandable way, maybe even with possible suggestions on what skills to practice.

    ### Considerations:
    - Obviously, which LLM do we use. Maybe see if there are smaller local models which are fine tuned to music analysis? We will need the hardware to run them though

## Other things that should be added
There are a lot of things that can be added to this to make it more useful and more powerful.

### 1. Optical Music Recognition

This software would be much more powerful if it could follow along your sheet music with you. There exists technology to convert pdf scans of sheet music into MIDI files, which computers can interact with. See [SheetVision](https://github.com/cal-pratt/SheetVision), which is completely free to use. 

The user can submit a pdf of their sheet music, which our app then converts to MIDI. It can then use a midi reading package in python (multiple exist), to follow along the music alongside the given recording and give more directed, specific feedback.

### 2. Practice sessions

Similar to how when you talk with LLMs such as GPT or Claude, they remember your chats across a conversation, our app should be able to remember recordings and the feedback it gave across a "practice session" consisting of multiple recording/feedback exchanges. That way, it is more informed on the progress the user is making and general areas of improvement across recordings.

To achieve this, we might need to research into how chatbots achieve this. Probably some short term memory in the form of a simple cache, may be very interesting to implement.

What else can be added?

## The overall application

Ultimately, if we ever get there, this software should be packaged into a mobile app, which allows users to easily submit recordings of them playing and get feedback. Some cool features could be:

- Like above, let users start practice sessions where they can have multiple consecutive recording/feedback echanges that the app is able to remember and build off of

- Allow users to create specific pieces that they are working on, so the app can remember user progress on the same piece across different practice sessions.

- Allow teachers / directors to track student progress, view summaries of student performance, and assign students things to practice through an interactive web page (with a very sophisticated and definitely way too hard to make backend)

Any other ideas?
