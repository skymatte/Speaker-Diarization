# - Kaldi ASR Report
    Name: Shivam Kumar 170668
![pipeline](https://i.ibb.co/726bsp3/kaldi-text-and-logo.png)

- - - - 

# Instructions
**NOTE:** 
Download all code and store **digits** folder in ***Documents/627/kaldi/egs/*** directory. 
    You can place remaining .py and .sh files any where but they should be together.
### Files & Folders
 |  Name |  Description |
 |---------|---------|
 | ```Transcribe.sh``` | Gives the prediction on test wav file |
 | ```Transcribe.py``` | Containes the backend for running all the bash scripts|
 | *digits* | Contains all the code for digit recognition |
 | ```dataprep.sh``` | Code to prepare datasets and Language folders |
 | *digits/data/test_original* | Contains the original test set files on which ER's are calculated [^footnote]|
 | *digits/digits_audio/test_original* | Contains the original test set audios on which ER's are calculated [^footnote] |
 [^footnote]: *-> May or may not be present (is created when transcribe file is run to save data from losing)
 
###### Note: To run any of the above code-
1. Replace the username *darkmask* in files ``data/*/wav.scp`` by your username {where * belongs to [test, train]}. 
2. Replace username **darkmask** in string of *```curr_path```* variable in line 7 & 68 in ``transcribe.py`` by your username.

### Instructions for training the code
- **For training the model** 
    > Change to digits directory (by using ```cd 627/kaldi/egs/digits/```)
    > ```chmod u+x run.sh```
    > ```./run.sh```
    > If error is being faced then run ```utils/fix_data_dir.sh data/test``` & ```utils/fix_data_dir.sh data/test```
    > Rerun ```./run.sh```
    

- **For obtaining transcriptions** 
    > Create a folder named __audios__ in *Downloads* directory.
    > Place all the test audio files (<file>.wav) in *audios* folder.
    > Change to 627 directory (by using  ```cd 627```)
    > Make the script executable by  ```chmod u+x transcribe.  ```
    > Suppose audio file is **file_name.wav**. Then we will pass only file_name as argument. (without .wav extention)
    > Run  ```./transcribe.sh *file_name*```
    > The predictions will be displayed at the end of output.

### Experiments done

Following experiments with code and dataset were conducted by me:
* I recorded datasets from each of my family members to created *variation* and *relatively large* dataset.
* I also used *global* speaker_id instead of dividing them into subgroups based on names and gender, so as to make the model to focus training just on the predictions. 
* Experimented by manually changing totgauss while Mono Training (also used Tidigits's script for better assumption)

##### Few Dependencies & installation process-

Install Kaldi in 627 directory by running:
1. `cd home/{your_username}/Documents/627/`
2. `git clone https://github.com/kaldi-asr/kaldi.git`
3. `cd kaldi/cd tools/`
4. `extras/check_dependencies.sh` (if something is not installed then install it using the comment provided in console.)
5. `make -j 4`
6. `extras/install_irstlm.sh`
7. `cd ../src/` and `run ./configure`
8. `make depend -j 4` and `make depend -j 4`

Kaldi is installed. Just we need to install one more dependency.
1. Install SRILM by running `kaldi/tools/install_srilm.sh` and following all the steps.
