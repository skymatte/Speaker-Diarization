# - Speaker Diarization


Group: Enthusiasm_Overflow


![pipeline](https://i.ibb.co/WvRNMNm/5fdfb94d16213785ea61b730-MPsoy-Uk-STj-Wm-Bd-Ql.png)

*

# Instructions

### Files

| Files & Links | Description |
| --- | --- |
| [ami_public_manual_1.6.2](https://drive.google.com/drive/folders/1bjxLF1i9prFotXZjB9brgCbpmEPqRzO7?usp=sharing) | Our training AMI CORPUS FOLDER containing speaker time stamps |
|[code](https://drive.google.com/drive/folders/1JQhDmVTyfLL-7-PfGfgtaZ25k1IFRHJ9?usp=sharing)|Contains python script |
|[ATML](https://drive.google.com/drive/folders/1WlfQSqm7KP7mNgWUVl1oL8XAWJbSFsZq?usp=sharing)|Folder containing ami_public_manual_1.6.2 and code folder|
| [amicorpusfinal](https://drive.google.com/drive/folders/1wphq5-rMTz2WC81Ma99YdGwESw4c5F7q?usp=sharing) | Training AMI WAV dataset. |


### Libraries needed to be imported

We use the following libraries:

* pydub
* xmltodict
* resemblyzer
* pyannote
* noisereduce
* spectralcluster
* PyTorch
* pyannote.metrics
* pyannote.core
* hdbscan
* keras
* tensorflow_addons
* python_speech_features

###### Note: To install any of the above libraries:

1. Use `pip install library_name` for your local system.
2. Use `!pip install library_name` when installing on Colab.
