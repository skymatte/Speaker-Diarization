import contextlib
import numpy as np
import wave
import librosa
import webrtcvad

# Reading wave file
def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

# Frame class
class Frame(object):
  def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

# generating frame for given duration and audio
def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(vad, frames, sample_rate):
    is_speech = []
    for frame in frames:
        is_speech.append(vad.is_speech(frame.bytes, sample_rate))
    return is_speech

# generating segments, used to seperate noise from speech
def vad(file):
    audio, sample_rate = read_wave(file)
    vad = webrtcvad.Vad(2)
    frames = frame_generator(10, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(vad, frames, sample_rate)
    return segments

# checking if given chunk contain voice or not, according to threshold limit
def speech(file,threshold):
  dummy = 0
  data = []
  segments = vad(file)
  if(threshold > segments.count(True)/len(segments)):
    return False
  return True

# return time segment of voice part from complete wav file
def fxn(file):
  segments = vad(file)
  segments = np.asarray(segments)
  
  dummy = 0.01*np.where(segments[:-1] != segments[1:])[0] +.01 
  print(segments,(segments[:-1] != segments[1:]).tolist().count(True))
  if len(dummy)%2==0:
    dummy = dummy
  else:
    dummy = np.delete(dummy, len(dummy)-1)
  print(len(dummy))
  voice = dummy.reshape(int(len(dummy)/2),2)
  
  return voice

# return time segment of voice part from complete wav file (similar to fxn function)
def allSegments(file):
  dummy = 0
  data = []
  startEndTime = []
  segments = vad(file)
  print(segments.count(True)/len(segments))
  for i in range(1,len(segments)):
    if(segments[i]):
      if(len(startEndTime)==0):
        startEndTime.append([i*10/1000,(i*10+10)/1000])
      else:
        if(segments[i-1]):
          startEndTime[-1][1]= (i*10+10)/1000
        else:
          startEndTime.append([i*10/1000,(i*10+10)/1000])
  return startEndTime
