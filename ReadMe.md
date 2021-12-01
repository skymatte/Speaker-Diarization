# - Speaker Diarization

    Group: Enthusiasm_Overflow
    Shivam Kumar 170668 
    Yash Mittal 170818 
    Prateek Varshney 170494


![pipeline](https://i.ibb.co/WvRNMNm/5fdfb94d16213785ea61b730-MPsoy-Uk-STj-Wm-Bd-Ql.png)


# Instructions for setting up Drive


Since we ran all our experiments on Google Colab, to reproduce our code the user will need to download the above data folders and upload them at the following locations (respectively) on their Google Drive:

| Folders to be downloaded | Description | Path at which to upload in your Google Drive|
| --- | :--------------: | :---:|
| [YashVAD](https://drive.google.com/drive/folders/16cib19M3i9xZ8MRiYyXwKS6Hy4Z5Rey3?usp=sharing), [CNN](https://drive.google.com/drive/folders/1M9otDYWznoDtcNBNQHzvGa9BYrL6Ar_M?usp=sharing), [TransferLearningBestModels](https://drive.google.com/drive/folders/1Pg22wZCBhSg-bmcNXt57rrP3145LZiwk?usp=sharing)| Folders containing Model weights |'/content/drive/MyDrive/' |
|[LSTM_keras_50epochs_completedata_nonfreeze_SGD.h5](https://drive.google.com/file/d/11Y3B2Fg5OyFh2X9LT0MCRGugptcHbDYT/view?usp=sharing) [LSTM_keras_50epochs_completedata_history_nofreeze_SGD](https://drive.google.com/file/d/1iyIJ1EPbWPpq1spjKdLgMHelA41ZfJEc/view?usp=sharing)|Saved Weights for Transfer Learning Variant 3|‘/content/drive/MyDrive/’|
|[ATML](https://drive.google.com/drive/folders/1WlfQSqm7KP7mNgWUVl1oL8XAWJbSFsZq?usp=sharing)|Folder containing ami_public_manual_1.6.2 and code folder|‘/content/drive/MyDrive/’|
| [amicorpusfinal](https://drive.google.com/drive/folders/1wphq5-rMTz2WC81Ma99YdGwESw4c5F7q?usp=sharing) | Training AMI WAV dataset. |‘/content/drive/MyDrive/’|
| [Hindi](https://drive.google.com/drive/folders/1XVBfXWWN-IlNCiniapw5CMQH6q2L-7YR?usp=sharing)| Constains dataset, model & python scripts for *Hindi_English BiLSTM* Model| "/content/drive/MyDrive/" |
|plots (create an empty folder)|create an empty folder named 'plots' to store generated plots|‘/content/drive/MyDrive/’|


# Discription of files present in this Github Repo.
## Main Project Codes
Contains the following jupyter notebooks:

| Files  | Description |
| --- | --- |
| *Resemblyser_spectral.ipynb* | Contains the baseline Speaker Diarization code which uses a pre-trained instance of their model (trained on fixed-length segmentsextracted from a large corpus) as the Embedding module for our Speaker Diarization system.  |
|*CNN_embedding_submission.ipynb*| Uses Mel-log spectrum and MFCC feature extractor as well as a denoiser to remove the silence parts and speech noise and a CNN Model to generate the embeddings|
| *AMI_LSTM_Submission_BaseLine.ipynb* | Uses the log-melspectrum of the wav chunks as the input vectors (features) to the LSTM based Embedding module. |
|*DER_Hindi_English.ipynb*|Contains code for Speaker Diarization using BiLSTM model trained on Hindi English Custom Dataset.|
| *vad_comparisons.ipynb*| Compares the performance of the three VAD methods: WebRTC-VAD, Voice Activity Detector, LSTM based Model|




## Transfer Learning Variants
Contains the following jupyter notebooks:
| Files  | Description |
| --- | --- |
| *Transfer_Learning_Variant1.ipynb*|  Passes the dataset to the pre-trained Hindi-English-BiLSTM and the resulting "refined" features to train a new Embedding Module from scratch. This is similar to passing the dataset through a sequence of 2 models aligned one after the other.|
|*Transfer_Learning_Variant2.ipynb*| Combines the above 2 models into one: freezes the weights of the BiLSTM layers of the Hindi-English-BiLSTM Model, removes and replaces the TimeDistributed Dense Layers with one LSTM + Simple Dense Layers and retrain the model using MFCC features of the AMI-Corpus Dataset, thereby enabling only the training of the top layers.
|*Transfer_Learning_Variant3.ipynb*| Similar to Variant 2 except that it also unfreezes the BiLSTM layers as well, i.e., trains the "pre-trained" model (after replacing the Dense Layers) end to end on the current dataset and finetunes it accordingly.|


<!-- ### Main Project Codes

| Files  | Description |
| --- | --- |
| [ami_public_manual_1.6.2](https://drive.google.com/drive/folders/1bjxLF1i9prFotXZjB9brgCbpmEPqRzO7?usp=sharing) | Our training AMI CORPUS FOLDER containing speaker time stamps |
|[code](https://drive.google.com/drive/folders/1JQhDmVTyfLL-7-PfGfgtaZ25k1IFRHJ9?usp=sharing)|Contains python script |
|[ATML](https://drive.google.com/drive/folders/1WlfQSqm7KP7mNgWUVl1oL8XAWJbSFsZq?usp=sharing)|Folder containing ami_public_manual_1.6.2 and code folder|
| [amicorpusfinal](https://drive.google.com/drive/folders/1wphq5-rMTz2WC81Ma99YdGwESw4c5F7q?usp=sharing) | Training AMI WAV dataset. | -->


## Demo
[**Demos_Part1**](https://drive.google.com/file/d/1COzn6TQkSmAFcyxyrxamStSbYqaxXnIg/view?usp=sharing) Contains the results of speaker diarization on live run on Youtube Clip. 

[**Demo_Part2**](https://drive.google.com/file/d/1VPlyNYHE_jPbmORbpPoBou-gfhcMoV-u/view?usp=sharing) Contains the results of our variation of applying transfer learning to adapt our model from one dataset to another dataset.

## Libraries needed to be imported

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

### Note: To install any of the above libraries:

1. Use `pip install library_name` for your local system.
2. Use `!pip install library_name` when installing on Colab.
