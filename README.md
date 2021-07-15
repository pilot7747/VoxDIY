# About

This repository provides data and code for *"CrowdSpeech and Vox DIY: Benchmark Dataset for Crowdsourced Audio Transcription"* paper.

The collected transcriptions stored in `data/*-crowd.tsv`, ground-truth transcriptions stored in `data/*-gt.txt`. We also provide a code for the annotation process 
and speech synthesis in `annotation` and `speech_sythesis` folders, respectively.

# Evaluation
First, you may need to install some dependencies:
```bash
pip3 install crowd-kit toloka-kit jiwer
```

Then, you can easily evaluate all our baseline aggregation methods by a single command:
```bash
python3 baselines.py data/<dataset>-gt.txt data/<dataset>-crowd.tsv
```

In order to get the **Oracle** result, run
```bash
python3 oracle.py data/<dataset>-gt.txt data/<dataset>-crowd.tsv
```

You can also get the *Inter-Rater Agreement* by running
```bash
python3 agreement.py data/<dataset>-crowd.tsv
```

# Annotation

You can find an IPython notebook with a code for the data collection process. For the quality control, we use a special class, `TaskProcessor`, that
gets all the submits that are not accepted or rejected at the moment, calculates workers' skills, and checks if a submit should be accepted or rejected.
# License

## Code

© YANDEX LLC, 2021. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.

## Data

© YANDEX LLC, 2021. Licensed under the Creative Commons Attribution 4.0 license. See data/LICENSE file for more details.

# Acknowledgements

[LibriSpeech](https://www.openslr.org/12) dataset is used under the Creative Commons Attribution 4.0 license.

[CrowdWSA2019](https://github.com/garfieldpigljy/CrowdWSA2019) dataset is used under the Creative Commons Attribution 4.0 license.
