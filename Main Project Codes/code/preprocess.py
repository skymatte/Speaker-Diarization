#@title
from pydub import AudioSegment
import xmltodict
import os
os.path.join('a','b')

from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

from operator import itemgetter
from lxml import etree
import xml.etree.ElementTree as ET
import json

import librosa
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio, display
from math import ceil

from pyannote.core import Segment, Timeline, Annotation, notebook
import noisereduce as nr
from tqdm import tqdm

from python_speech_features import fbank
from random import choice

NUM_FBANKS = 64

## parsing transcript xml file
def parse_ami_transcript_xml(file, speaker_id):
    """
    Parsing AMI transcript XML file
    """
    xmlp = ET.XMLParser(encoding="ISO-8859-1")
    f = ET.parse(file, parser=xmlp)
    root = f.getroot()
    transcript = []
    for element in list(root):
        if element.tag == 'w':
            if element.text:
                text = element.text
            else:
                text = ''
            if element.get('punc'):
                punc = True
            else:
                punc = False

            transcript.append({
                'start': float(element.get('starttime')),
                'end': float(element.get('endtime')),
                'text': text,
                'punc': punc,
                'speaker_id': speaker_id
            })
    previous = transcript[0]
    for index, elem in enumerate(transcript):
        if elem['start'] == previous['end']:
            previous['end'] = elem['end']
            previous['text'] += ('' if elem['punc'] else ' ') + elem['text']
            transcript[index] = None
        else:
            del elem['punc']
            previous = elem
    transcript = [t for t in transcript if t and len(t['text'])]
    return transcript

## Splitting wave file according to t1 and t2
def split_wav(directory_path,filename,t1,t2, chunk_id):
  t1=float(t1)*1000
  t2=float(t2)*1000
  complete_file_path = os.path.join(directory_path, filename)
  audio_file = AudioSegment.from_wav(complete_file_path)
  audio_file = audio_file[t1:t2]
  audio_file.export(complete_file_path.replace('.wav','')+f'_{chunk_id}.wav', format = 'wav')

## Parsing xml file
def parse_xml(directory_path, filename):
  complete_file_path = os.path.join(directory_path, filename)
  content = open(complete_file_path).read()
  return xmltodict.parse(content)

#@title
## reading mfcc
def read_mfcc(input_filename, sample_rate):
    audio = Audio.read(input_filename, sample_rate)
    energy = np.abs(audio)
    silence_threshold = np.percentile(energy, 95)
    offsets = np.where(energy > silence_threshold)[0]
    
    # left_blank_duration_ms = (1000.0 * offsets[0]) // self.sample_rate  # frame_id to duration (ms)
    # right_blank_duration_ms = (1000.0 * (len(audio) - offsets[-1])) // self.sample_rate

    audio_voice_only = audio[offsets[0]:offsets[-1]]
    mfcc = mfcc_fbank(audio_voice_only, sample_rate)
    return mfcc


## Audio class for preprocessing
class Audio:

    def __init__(self, cache_dir: str, audio_dir: str = None, sample_rate: int = SAMPLE_RATE, ext='flac'):
        self.ext = ext
        self.cache_dir = os.path.join(cache_dir, 'audio-fbanks')
        ensures_dir(self.cache_dir)
        if audio_dir is not None:
            self.build_cache(os.path.expanduser(audio_dir), sample_rate)
        self.speakers_to_utterances = defaultdict(dict)
        for cache_file in find_files(self.cache_dir, ext='npy'):
            # /path/to/speaker_utterance.npy
            speaker_id, utterance_id = Path(cache_file).stem.split('_')
            self.speakers_to_utterances[speaker_id][utterance_id] = cache_file

    @property
    def speaker_ids(self):
        return sorted(self.speakers_to_utterances)

    @staticmethod
    def trim_silence(audio, threshold):
        """Removes silence at the beginning and end of a sample."""
        energy = librosa.feature.rms(audio)
        frames = np.nonzero(np.array(energy > threshold))
        indices = librosa.core.frames_to_samples(frames)[1]

        # Note: indices can be an empty array, if the whole audio was silence.
        audio_trim = audio[0:0]
        left_blank = audio[0:0]
        right_blank = audio[0:0]
        if indices.size:
            audio_trim = audio[indices[0]:indices[-1]]
            left_blank = audio[:indices[0]]  # slice before.
            right_blank = audio[indices[-1]:]  # slice after.
        return audio_trim, left_blank, right_blank

    @staticmethod
    def read(filename, sample_rate=SAMPLE_RATE):
        audio, sr = librosa.load(filename, sr=sample_rate, mono=True, dtype=np.float32)
        assert sr == sample_rate
        return audio

    def build_cache(self, audio_dir, sample_rate):
        logger.info(f'audio_dir: {audio_dir}.')
        logger.info(f'sample_rate: {sample_rate:,} hz.')
        audio_files = find_files(audio_dir, ext=self.ext)
        audio_files_count = len(audio_files)
        assert audio_files_count != 0, f'Could not find any {self.ext} files in {audio_dir}.'
        logger.info(f'Found {audio_files_count:,} files in {audio_dir}.')
        with tqdm(audio_files) as bar:
            for audio_filename in bar:
                bar.set_description(audio_filename)
                self.cache_audio_file(audio_filename, sample_rate)

    def cache_audio_file(self, input_filename, sample_rate):
        sp, utt = extract_speaker_and_utterance_ids(input_filename)
        cache_filename = os.path.join(self.cache_dir, f'{sp}_{utt}.npy')
        if not os.path.isfile(cache_filename):
            try:
                mfcc = read_mfcc(input_filename, sample_rate)
                np.save(cache_filename, mfcc)
            except librosa.util.exceptions.ParameterError as e:
                logger.error(e)


## padding mfcc
def pad_mfcc(mfcc, max_length):  # num_frames, nfilt=64.
    if len(mfcc) < max_length:
        mfcc = np.vstack((mfcc, np.tile(np.zeros(mfcc.shape[1]), (max_length - len(mfcc), 1))))
    return mfcc

## Generating fbanks and energies array
def mfcc_fbank(signal: np.array, sample_rate: int):  # 1D signal array.
    # Returns MFCC with shape (num_frames, n_filters, 3).
    filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=NUM_FBANKS)
    frames_features = normalize_frames(filter_banks)
    # delta_1 = delta(filter_banks, N=1)
    # delta_2 = delta(delta_1, N=1)
    # frames_features = np.transpose(np.stack([filter_banks, delta_1, delta_2]), (1, 2, 0))
    return np.array(frames_features, dtype=np.float32)  # Float32 precision is enough here.


## Normalizing
def normalize_frames(m, epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]

## sampling from mfcc
def sample_from_mfcc(mfcc, max_length):
    if mfcc.shape[0] >= max_length:
        r = choice(range(0, len(mfcc) - max_length + 1))
        s = mfcc[r:r + max_length]
    else:
        s = pad_mfcc(mfcc, max_length)
    return np.expand_dims(s, axis=-1)

## sampling from mfcc file
def sample_from_mfcc_file(utterance_file, max_length):
    mfcc = np.load(utterance_file)
    return sample_from_mfcc(mfcc, max_length)