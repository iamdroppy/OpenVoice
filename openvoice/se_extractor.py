import os
import glob
import torch
import hashlib
import librosa
import base64
from glob import glob
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel
import hashlib
import base64
import librosa
from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments
from loguru import logger

class WhisperConfig:
    _model_size = "medium"
    _model = None
    _compute_type = "float16"
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, model_size="medium", device=None, compute_type="float16"):
        self._model_size = model_size
        self._compute_type = compute_type
        if device is not None:
            self._device = device
        self._model = None

    def model_size(self):
        return self._model_size
    
    def model(self):
        if self._model is None:
            self._model = WhisperModel(self._model_size, device=self._device, compute_type=self._compute_type)
        return self._model

    def device(self):
        return self._device
    
    def compute_type(self):
        return self._compute_type
    
    def set_model_size(self, size):
        if size != self._model_size:
            self._model_size = size
            self._model = None
    
    def set_compute_type(self, compute_type):
        if compute_type != self._compute_type:
            self._compute_type = compute_type
            self._model = None

def split_audio_whisper(audio_path, audio_name, target_dir='processed', whisper_size='medium', whisper_ctype='float16'):
    whisper_info: WhisperConfig = WhisperConfig(model_size=whisper_size, compute_type=whisper_ctype)

    model = whisper_info.model()
    if model is None:
        model = WhisperModel(whisper_info.model_size(), device=whisper_info.device(), compute_type=whisper_info.compute_type())
    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)

    target_folder = os.path.join(target_dir, audio_name)
    
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)    

    # create directory
    os.makedirs(target_folder, exist_ok=True)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)

    # segments
    s_ind = 0
    start_time = None
    
    logger.trace(f"Whisper detected {len(segments)} segments")
    for k, w in enumerate(segments):
        # process with the time
        if k == 0:
            start_time = max(0, w.start)

        end_time = w.end
        # get time
        total_time = w.end - w.start
        # calculate confidence
        if len(w.words) > 0:
            confidence = sum([s.probability for s in w.words]) / len(w.words)
        else:
            confidence = 0.
        # clean text
        text = w.text.replace('...', '')
        logger.trace(f"Segment {k}: start {w.start}, end {w.end}, total {total_time}, confidence: {confidence}, text: {w.text}")

        # left 0.08s for each audios
        audio_seg = audio[int( start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]

        # segment file name
        fname = f"{audio_name}_seg{s_ind}.wav"

        # filter out the segment shorter than 1.5s and longer than 20s
        save = audio_seg.duration_seconds > 1.5 and \
                audio_seg.duration_seconds < 20. and \
                len(text) >= 2 and len(text) < 200 

        if save:
            output_file = os.path.join(wavs_folder, fname)
            audio_seg.export(output_file, format='wav')

        if k < len(segments) - 1:
            start_time = max(0, segments[k+1].start - 0.08)

        s_ind = s_ind + 1
    return wavs_folder


def split_audio_vad(audio_path, audio_name, target_dir, split_seconds=10.0):
    SAMPLE_RATE = 16000
    audio_vad = get_audio_tensor(audio_path)
    segments = get_vad_segments(
        audio_vad,
        output_sample=True,
        min_speech_duration=0.1,
        min_silence_duration=1,
        method="silero",
    )
    segments = [(seg["start"], seg["end"]) for seg in segments]
    segments = [(float(s) / SAMPLE_RATE, float(e) / SAMPLE_RATE) for s,e in segments]
    logger.debug(f"Loaded VAD segments: {segments}")
    audio_active = AudioSegment.silent(duration=0)
    audio = AudioSegment.from_file(audio_path)

    for start_time, end_time in segments:
        audio_active += audio[int( start_time * 1000) : int(end_time * 1000)]
    
    audio_dur = audio_active.duration_seconds
    logger.info(f'Audio duration after vad: {audio_dur}')
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)
    start_time = 0.
    count = 0
    num_splits = int(np.round(audio_dur / split_seconds))
    assert num_splits > 0, 'input audio is too short'
    interval = audio_dur / num_splits

    for i in range(num_splits):
        end_time = min(start_time + interval, audio_dur)
        if i == num_splits - 1:
            end_time = audio_dur
        output_file = f"{wavs_folder}/{audio_name}_seg{count}.wav"
        audio_seg = audio_active[int(start_time * 1000): int(end_time * 1000)]
        audio_seg.export(output_file, format='wav')
        start_time = end_time
        count += 1
    return wavs_folder

def hash_numpy_array(audio_path):
    array, _ = librosa.load(audio_path, sr=None, mono=True)
    # Convert the array to bytes
    array_bytes = array.tobytes()
    # Calculate the hash of the array bytes
    hash_object = hashlib.sha256(array_bytes)
    hash_value = hash_object.digest()
    # Convert the hash value to base64
    base64_value = base64.b64encode(hash_value)
    return base64_value.decode('utf-8')[:16].replace('/', '_^')
def get_whisper_model_size():
    return model_size
def set_whisper_model_size(size):
    global model_size
    model_size = size
def get_se(audio_path, vc_model, target_dir='processed', vad=True, whisper_size='medium', whisper_ctype='float16'):
    device = vc_model.device
    version = vc_model.version
    logger.info("OpenVoice version:" + version)

    audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{version}_{hash_numpy_array(audio_path)}"
    se_path = os.path.join(target_dir, audio_name, 'se.pth')

    # if os.path.isfile(se_path):
    #     se = torch.load(se_path).to(device)
    #     return se, audio_name
    # if os.path.isdir(audio_path):
    #     wavs_folder = audio_path
    
    if vad:
        wavs_folder = split_audio_vad(audio_path, target_dir=target_dir, audio_name=audio_name)
    else:
        wavs_folder = split_audio_whisper(audio_path, target_dir=target_dir, audio_name=audio_name, whisper_size=whisper_size, whisper_ctype=whisper_ctype)
    
    audio_segs = glob(f'{wavs_folder}/*.wav')
    if len(audio_segs) == 0:
        raise NotImplementedError('No audio segments found!')
    
    return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name

