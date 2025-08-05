"""
python app_pruna.py     --task multitalk-14B     --size multitalk-480     --frame_num 81     
--mode streaming     --ckpt_dir /weights/Wan2.1-I2V-14B-480P     
--wav2vec_dir /weights/chinese-wav2vec2-base     
--lora_dir /weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors    
--input_json /examples/single_example_1.json     --save_file outputs/final_targeted_pruna_run    
--benchmark     --use_pruna     --use_teacache --teacache_thresh 0.35     --use_apg     --offload_model False    
--sample_steps 8     --sample_shift 2
"""


import argparse
import logging
import os
import sys
import json
import warnings
from datetime import datetime
import time
from contextlib import contextmanager
from functools import wraps
warnings.filterwarnings('ignore')
import random
import torch
import torch.distributed as dist
from PIL import Image
import subprocess
import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool
from wan.utils.multitalk_utils import save_video_ffmpeg
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf
import re

# Try importing Pruna Pro
try:
    # Use the Pruna Pro library
    from pruna_pro import SmashConfig, smash
    PRUNA_AVAILABLE = True
except ImportError:
    PRUNA_AVAILABLE = False
    # Update the warning message
    print("Warning: Pruna Pro not installed. Run 'pip install pruna_pro' for optimizations.")

# Global timing storage
TIMING_ENABLED = False
TIMING_STACK = []
TIMING_RESULTS = {}

@contextmanager
def timing_context(name):
    """Context manager for timing code blocks"""
    if not TIMING_ENABLED:
        yield
        return
    
    start_time = time.time()
    indent = "  " * len(TIMING_STACK)
    TIMING_STACK.append(name)
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        full_name = " > ".join(TIMING_STACK)
        TIMING_RESULTS[full_name] = elapsed
        TIMING_STACK.pop()
        
        # Log immediately for real-time feedback
        logging.info(f"{indent}[TIMING] {name}: {elapsed:.3f}s")

def timing_decorator(func):
    """Decorator for timing functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TIMING_ENABLED:
            return func(*args, **kwargs)
        
        with timing_context(f"Function: {func.__name__}"):
            return func(*args, **kwargs)
    
    return wrapper

def log_timing_summary():
    """Print a summary of all timing results"""
    if not TIMING_ENABLED or not TIMING_RESULTS:
        return
    
    logging.info("\n" + "="*80)
    logging.info("TIMING SUMMARY")
    logging.info("="*80)
    
    # Sort by elapsed time
    sorted_results = sorted(TIMING_RESULTS.items(), key=lambda x: x[1], reverse=True)
    
    for name, elapsed in sorted_results:
        depth = name.count(" > ")
        indent = "  " * depth
        logging.info(f"{indent}{name}: {elapsed:.3f}s")
    
    logging.info("="*80 + "\n")

def validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    
    # Auto-enable FP8 if quant_dir is provided but quant is not set
    if args.quant_dir and not args.quant:
        args.quant = "fp8"
        logging.info(f"Auto-enabled {args.quant} quantization based on quant_dir")
    
    # Set default steps based on LoRA usage
    if args.sample_steps is None:
        if args.lora_dir:
            args.sample_steps = 8  # Fewer steps with LoRA
        else:
            args.sample_steps = 30 if args.use_teacache else 40
    
    if args.sample_shift is None:
        if args.size == 'multitalk-480':
            args.sample_shift = 3 if args.lora_dir else 5 if args.use_teacache else 7
        elif args.size == 'multitalk-720':
            args.sample_shift = 5 if args.lora_dir else 8 if args.use_teacache else 11
        else:
            raise NotImplementedError(f'Not supported size')
    
    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, 99999999)
    
    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="multitalk-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="multitalk-480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The buckget size of the generated video. The aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to be generated in one clip. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the Wan checkpoint directory.")
    parser.add_argument(
        "--quant_dir",
        type=str,
        default=None,
        help="The path to the Wan quant checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default=None,
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--lora_dir",
        type=str,
        nargs='+',
        default=None,
        help="The paths to the LoRA checkpoint files."
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        nargs='+',
        default=[1.2],
        help="Controls how much to influence the outputs with the LoRA parameters. Accepts multiple float values."
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--audio_save_dir",
        type=str,
        default='save_audio',
        help="The path to save the audio embedding.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--input_json",
        type=str,
        default='examples.json',
        help="[meta file] The condition path to generate the video.")
    parser.add_argument(
        "--motion_frame",
        type=int,
        default=25,
        help="Driven frame length used in the mode of long video genration.")
    parser.add_argument(
        "--mode",
        type=str,
        default="clip",
        choices=['clip', 'streaming'],
        help="clip: generate one video chunk, streaming: long video generation")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_text_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale for text control.")
    parser.add_argument(
        "--sample_audio_guide_scale",
        type=float,
        default=4.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        required=False,
        help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
    )
    parser.add_argument(
        "--audio_mode",
        type=str,
        default="localfile",
        choices=['localfile', 'tts'],
        help="localfile: audio from local wav file, tts: audio from TTS")
    parser.add_argument(
        "--use_teacache",
        action="store_true",
        default=False,
        help="Enable teacache for video generation."
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Threshold for teacache."
    )
    parser.add_argument(
        "--use_apg",
        action="store_true",
        default=False,
        help="Enable adaptive projected guidance for video generation (APG)."
    )
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=-0.75,
        help="Momentum used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=55,
        help="Norm threshold used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--color_correction_strength",
        type=float,
        default=1.0,
        help="strength for color correction [0.0 -- 1.0]."
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Quantization type, must be 'int8' or 'fp8'."
    )
    
    # Pruna specific arguments
    parser.add_argument(
        "--use_pruna",
        action="store_true",
        default=False,
        help="Enable Pruna AI optimization (torch.compile)."
    )
    
    parser.add_argument(
        "--debug_timing",
        action="store_true",
        default=False,
        help="Enable detailed timing logs for debugging performance."
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run in benchmark mode with performance summary."
    )
    
    args = parser.parse_args()
    validate_args(args)
    return args

@timing_decorator
def custom_init(device, wav2vec):    
    with timing_context("Load Wav2Vec2 Model"):
        audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    
    with timing_context("Freeze Feature Extractor"):
        audio_encoder.feature_extractor._freeze_parameters()
    
    with timing_context("Load Wav2Vec2 Feature Extractor"):
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    
    return wav2vec_feature_extractor, audio_encoder

@timing_decorator
def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

@timing_decorator
def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    with timing_context("Load Audio Files"):
        if not (left_path=='None' or right_path=='None'):
            with timing_context("Load Both Audio Files"):
                human_speech_array1 = audio_prepare_single(left_path)
                human_speech_array2 = audio_prepare_single(right_path)
        elif left_path=='None':
            with timing_context("Load Right Audio Only"):
                human_speech_array2 = audio_prepare_single(right_path)
                human_speech_array1 = np.zeros(human_speech_array2.shape[0])
        elif right_path=='None':
            with timing_context("Load Left Audio Only"):
                human_speech_array1 = audio_prepare_single(left_path)
                human_speech_array2 = np.zeros(human_speech_array1.shape[0])
    
    with timing_context("Process Audio Type"):
        if audio_type=='para':
            new_human_speech1 = human_speech_array1
            new_human_speech2 = human_speech_array2
        elif audio_type=='add':
            new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
            new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    
    with timing_context("Sum Audio Arrays"):
        sum_human_speechs = new_human_speech1 + new_human_speech2
    
    return new_human_speech1, new_human_speech2, sum_human_speechs

def init_logging(rank):
    """Configure global logging."""
    root_logger = logging.getLogger()
    # Remove handlers that might have been added by previously imported modules
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    # Decide level according to rank
    log_level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
        force=True,  # Python>=3.8 ensures reconfiguration even if already set
    )

@timing_decorator
def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    with timing_context("Audio Duration Calculation"):
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25 # Assume the video fps is 25
    
    # wav2vec_feature_extractor
    with timing_context("Wav2Vec Feature Extraction"):
        audio_feature = np.squeeze(
            wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)
    
    # audio encoder
    with timing_context("Audio Encoding"):
        with torch.no_grad():
            embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
    
    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None
    
    with timing_context("Embedding Post-processing"):
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        audio_emb = audio_emb.cpu().detach()
    
    return audio_emb

@timing_decorator
def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    with timing_context("FFmpeg Audio Extraction"):
        subprocess.run(ffmpeg_command, check=True)
    
    with timing_context("Load Audio with Librosa"):
        human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    
    with timing_context("Loudness Normalization"):
        human_speech_array = loudness_norm(human_speech_array, sr)
    
    with timing_context("Cleanup Temp File"):
        os.remove(raw_audio_path)
    return human_speech_array

@timing_decorator
def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array

@timing_decorator
def process_tts_single(text, save_dir, voice1):    
    s1_sentences = []
    with timing_context("Initialize TTS Pipeline"):
        pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')
    with timing_context("Load Voice Tensor"):
        voice_tensor = torch.load(voice1, weights_only=True)
    
    with timing_context("TTS Generation"):
        generator = pipeline(
            text, voice=voice_tensor, # <= change voice here
            speed=1, split_pattern=r'\n+'
        )
        audios = []
        for i, (gs, ps, audio) in enumerate(generator):
            with timing_context(f"TTS Chunk {i}"):
                audios.append(audio)
    
    with timing_context("Concatenate Audio Chunks"):
        audios = torch.concat(audios, dim=0)
        s1_sentences.append(audios)
        s1_sentences = torch.concat(s1_sentences, dim=0)
    
    save_path1 =f'{save_dir}/s1.wav'
    with timing_context("Save TTS Audio"):
        sf.write(save_path1, s1_sentences, 24000) # save each audio file
    
    with timing_context("Reload Audio at 16kHz"):
        s1, _ = librosa.load(save_path1, sr=16000)
    
    return s1, save_path1

@timing_decorator
def process_tts_multi(text, save_dir, voice1, voice2):
    with timing_context("Parse Multi-Speaker Text"):
        pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
        matches = re.findall(pattern, text, re.DOTALL)
    
    s1_sentences = []
    s2_sentences = []
    with timing_context("Initialize TTS Pipeline"):
        pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')
    
    for idx, (speaker, content) in enumerate(matches):
        with timing_context(f"Processing Speaker {speaker} - Segment {idx}"):
            if speaker == '1':
                with timing_context("Load Voice 1"):
                    voice_tensor = torch.load(voice1, weights_only=True)
                with timing_context("Generate Speech for Speaker 1"):
                    generator = pipeline(
                        content, voice=voice_tensor, # <= change voice here
                        speed=1, split_pattern=r'\n+'
                    )
                    audios = []
                    for i, (gs, ps, audio) in enumerate(generator):
                        audios.append(audio)
                with timing_context("Concatenate and Append Speaker 1 Audio"):
                    audios = torch.concat(audios, dim=0)
                    s1_sentences.append(audios)
                    s2_sentences.append(torch.zeros_like(audios))
            elif speaker == '2':
                with timing_context("Load Voice 2"):
                    voice_tensor = torch.load(voice2, weights_only=True)
                with timing_context("Generate Speech for Speaker 2"):
                    generator = pipeline(
                        content, voice=voice_tensor, # <= change voice here
                        speed=1, split_pattern=r'\n+'
                    )
                    audios = []
                    for i, (gs, ps, audio) in enumerate(generator):
                        audios.append(audio)
                with timing_context("Concatenate and Append Speaker 2 Audio"):
                    audios = torch.concat(audios, dim=0)
                    s2_sentences.append(audios)
                    s1_sentences.append(torch.zeros_like(audios))
    
    with timing_context("Finalize Multi-Speaker Audio"):
        s1_sentences = torch.concat(s1_sentences, dim=0)
        s2_sentences = torch.concat(s2_sentences, dim=0)
        sum_sentences = s1_sentences + s2_sentences
        
    save_path1 =f'{save_dir}/s1.wav'
    save_path2 =f'{save_dir}/s2.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    
    with timing_context("Save Multi-Speaker Audio Files"):
        sf.write(save_path1, s1_sentences, 24000) # save each audio file
        sf.write(save_path2, s2_sentences, 24000)
        sf.write(save_path_sum, sum_sentences, 24000)
    
    with timing_context("Reload Audio at 16kHz"):
        s1, _ = librosa.load(save_path1, sr=16000)
        s2, _ = librosa.load(save_path2, sr=16000)
    
    return s1, s2, save_path_sum

def generate(args):
    # Enable global timing if requested
    global TIMING_ENABLED
    TIMING_ENABLED = args.debug_timing or args.benchmark
    import torch
    
    # 1. Enable TF32 for performance boost on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    # Track benchmark start time
    if args.benchmark:
        benchmark_start = time.time()
    
    with timing_context("Total Pipeline Execution"):
        with timing_context("Initialization"):
            rank = int(os.getenv("RANK", 0))
            world_size = int(os.getenv("WORLD_SIZE", 1))
            local_rank = int(os.getenv("LOCAL_RANK", 0))
            device = local_rank
            init_logging(rank)
            
            # Log optimization settings
            if rank == 0:
                logging.info("\n" + "="*60)
                logging.info("OPTIMIZATION SETTINGS")
                logging.info("="*60)
                logging.info(f"TeaCache: {args.use_teacache} (thresh: {args.teacache_thresh})")
                logging.info(f"APG: {args.use_apg}")
                logging.info(f"Quantization: {args.quant or 'None'}")
                logging.info(f"LoRA: {'Yes' if args.lora_dir else 'No'}")
                logging.info(f"Sample Steps: {args.sample_steps}")
                logging.info(f"Sample Shift: {args.sample_shift}")
                if args.use_pruna and PRUNA_AVAILABLE:
                    logging.info("Pruna Pro: Yes (Using Torch Compiler)")
                else:
                    logging.info(f"Pruna: No")
                logging.info("="*60 + "\n")
            
            if args.offload_model is None:
                args.offload_model = False if world_size > 1 else True
                logging.info(f"offload_model is not specified, set to {args.offload_model}.")
            
            with timing_context("Distributed Setup"):
                if world_size > 1:
                    torch.cuda.set_device(local_rank)
                    dist.init_process_group(
                        backend="nccl",
                        init_method="env://",
                        rank=rank,
                        world_size=world_size)
                else:
                    assert not (
                        args.t5_fsdp or args.dit_fsdp
                    ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
                    assert not (
                        args.ulysses_size > 1 or args.ring_size > 1
                    ), f"context parallel are not supported in non-distributed environments."
                
                if args.ulysses_size > 1 or args.ring_size > 1:
                    assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
                    from xfuser.core.distributed import (
                        init_distributed_environment,
                        initialize_model_parallel,
                    )
                    init_distributed_environment(
                        rank=dist.get_rank(), world_size=dist.get_world_size())
                    initialize_model_parallel(
                        sequence_parallel_degree=dist.get_world_size(),
                        ring_degree=args.ring_size,
                        ulysses_degree=args.ulysses_size,
                    )
            
            cfg = WAN_CONFIGS[args.task]
            if args.ulysses_size > 1:
                assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
            
            logging.info(f"Generation job args: {args}")
            logging.info(f"Generation model config: {cfg}")
            
            if dist.is_initialized():
                base_seed = [args.base_seed] if rank == 0 else [None]
                dist.broadcast_object_list(base_seed, src=0)
                args.base_seed = base_seed[0]
            
            assert args.task == "multitalk-14B", 'You should choose multitalk in args.task.'
        
        with timing_context("Loading Input Data"):
            with open(args.input_json, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
        
        with timing_context("Audio Processing"):
            with timing_context("Initialize Audio Models"):
                wav2vec_feature_extractor, audio_encoder = custom_init('cpu', args.wav2vec_dir)
                # NOTE: We skip Pruna optimization for audio_encoder as it's not compatible
                # and only takes ~1s anyway - not worth optimizing
            
            args.audio_save_dir = os.path.join(args.audio_save_dir, input_data['cond_image'].split('/')[-1].split('.')[0])
            os.makedirs(args.audio_save_dir,exist_ok=True)
            
            if args.audio_mode=='localfile':
                with timing_context("Local Audio File Processing"):
                    if len(input_data['cond_audio'])==2:
                        with timing_context("Dual Audio Processing"):
                            new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(input_data['cond_audio']['person1'], input_data['cond_audio']['person2'], input_data['audio_type'])
                            with timing_context("Audio Embeddings Generation"):
                                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                                audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
                            with timing_context("Saving Audio Files"):
                                emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                                emb2_path = os.path.join(args.audio_save_dir, '2.pt')
                                sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
                                sf.write(sum_audio, sum_human_speechs, 16000)
                                torch.save(audio_embedding_1, emb1_path)
                                torch.save(audio_embedding_2, emb2_path)
                            input_data['cond_audio']['person1'] = emb1_path
                            input_data['cond_audio']['person2'] = emb2_path
                            input_data['video_audio'] = sum_audio
                    elif len(input_data['cond_audio'])==1:
                        with timing_context("Single Audio Processing"):
                            human_speech = audio_prepare_single(input_data['cond_audio']['person1'])
                            with timing_context("Audio Embedding Generation"):
                                audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
                            with timing_context("Saving Audio Files"):
                                emb_path = os.path.join(args.audio_save_dir, '1.pt')
                                sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
                                sf.write(sum_audio, human_speech, 16000)
                                torch.save(audio_embedding, emb_path)
                            input_data['cond_audio']['person1'] = emb_path
                            input_data['video_audio'] = sum_audio
            elif args.audio_mode=='tts':
                with timing_context("TTS Audio Processing"):
                    if 'human2_voice' not in input_data['tts_audio'].keys():
                        with timing_context("Single Voice TTS"):
                            new_human_speech1, sum_audio = process_tts_single(input_data['tts_audio']['text'], args.audio_save_dir, input_data['tts_audio']['human1_voice'])
                            with timing_context("TTS Audio Embedding"):
                                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                            with timing_context("Saving TTS Audio"):
                                emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                                torch.save(audio_embedding_1, emb1_path)
                            input_data['cond_audio']['person1'] = emb1_path
                            input_data['video_audio'] = sum_audio
                    else:
                        with timing_context("Dual Voice TTS"):
                            new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(input_data['tts_audio']['text'], args.audio_save_dir, input_data['tts_audio']['human1_voice'], input_data['tts_audio']['human2_voice'])
                            with timing_context("TTS Audio Embeddings"):
                                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                                audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
                            with timing_context("Saving TTS Audio"):
                                emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                                emb2_path = os.path.join(args.audio_save_dir, '2.pt')
                                torch.save(audio_embedding_1, emb1_path)
                                torch.save(audio_embedding_2, emb2_path)
                            input_data['cond_audio']['person1'] = emb1_path
                            input_data['cond_audio']['person2'] = emb2_path
                            input_data['video_audio'] = sum_audio
        
        with timing_context("MultiTalk Pipeline Creation"):
            logging.info("Creating MultiTalk pipeline.")
            wan_i2v = wan.MultiTalkPipeline(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                quant_dir=args.quant_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp, 
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),  
                t5_cpu=args.t5_cpu,
                lora_dir=args.lora_dir,
                lora_scales=args.lora_scale,
                quant=args.quant
            )
            
            # Apply Pruna optimization to the pipeline
            if args.use_pruna and PRUNA_AVAILABLE:
                with timing_context("Pruna Pipeline Optimization"):
                    logging.info("Optimizing MultiTalk pipeline with Pruna...")

                    # --- WorkAround for no-op compiler as a patch now.: Manually compile, then smash (no-op compiler) ---
                    # Pruna's SmashConfig does not expose all torch.compile flags,
                    # so we compile the model directly with the required arguments
                    # (`fullgraph=False`, `dynamic=True`) and then disable Pruna's
                    # internal compiler to avoid conflicts or redundant work.
                    
                    logging.info("Manually compiling the core DiT model with torch.compile...")
                    wan_i2v.model = torch.compile(wan_i2v.model, fullgraph=False, dynamic=True)
                    logging.info("Core model compiled successfully.")

                    # Now, "smash" the model, but with Pruna's compiler disabled.
                    # This would still allow other Pruna features (like quantization) to run if enabled.
                    smash_config = SmashConfig()
                    smash_config["compiler"] = None # IMPORTANT: Disable Pruna's compiler
                    smash_config["device"] = "cuda"
                    
                    try:
                        logging.info("Applying other Pruna optimizations (if any)...")
                        
                        # This call will now skip compilation but apply other optimizations.
                        wan_i2v.model = smash(
                            model=wan_i2v.model,
                            token=os.getenv("PRUNA_TOKEN", "sub_1RsdG4068nww4z9saaGpB1rw"), 
                            smash_config=smash_config,
                        )
                        
                        logging.info("Pruna optimizations applied successfully.")
                        
                    except Exception as e:
                        logging.warning(f"Pruna smash call failed: {e}")
                        logging.info("Continuing with only the manual torch.compile optimization.")
            
            if args.num_persistent_param_in_dit is not None:
                with timing_context("VRAM Management Setup"):
                    wan_i2v.vram_management = True
                    wan_i2v.enable_vram_management(
                        num_persistent_param_in_dit=args.num_persistent_param_in_dit
                    )
        
        with timing_context("Video Generation"):
            logging.info("Generating video ...")
            
            # 3. Update deprecated autocast call
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                video = wan_i2v.generate(
                    input_data,
                    size_buckget=args.size,
                    motion_frame=args.motion_frame,
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sampling_steps=args.sample_steps,
                    text_guide_scale=args.sample_text_guide_scale,
                    audio_guide_scale=args.sample_audio_guide_scale,
                    seed=args.base_seed,
                    offload_model=args.offload_model,
                    max_frames_num=args.frame_num if args.mode == 'clip' else 1000,
                    color_correction_strength = args.color_correction_strength,
                    extra_args=args,
                )
        
        if rank == 0:
            with timing_context("Video Saving"):
                if args.save_file is None:
                    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
                    pruna_suffix = "_pruna" if (args.use_pruna and PRUNA_AVAILABLE) else ""
                    args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}{pruna_suffix}"
                
                logging.info(f"Saving generated video to {args.save_file}.mp4")
                save_video_ffmpeg(video, args.save_file, [input_data['video_audio']], high_quality_save=False)
        
        logging.info("Finished.")
    
    # Benchmark summary if requested
    if args.benchmark and rank == 0:
        total_time = time.time() - benchmark_start
        
        # Extract key timings
        generation_time = TIMING_RESULTS.get("Total Pipeline Execution > Video Generation", 0)
        model_load_time = TIMING_RESULTS.get("Total Pipeline Execution > MultiTalk Pipeline Creation", 0)
        audio_time = TIMING_RESULTS.get("Total Pipeline Execution > Audio Processing", 0)
        
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        print(f"Total Time: {total_time:.2f}s")
        print(f"  Model Loading: {model_load_time:.1f}s")
        print(f"  Audio Processing: {audio_time:.1f}s") 
        print(f"  Video Generation: {generation_time:.1f}s")
        print(f"Frames Generated: {args.frame_num}")
        print(f"Time per Frame: {generation_time/args.frame_num:.2f}s")
        print(f"Target <60s: {'✓ PASSED' if total_time < 60 else '✗ FAILED'}")
        
        # Extrapolation for 81 frames if different
        if args.frame_num != 81:
            extrapolated_gen = (generation_time / args.frame_num * 81)
            extrapolated_total = model_load_time + audio_time + extrapolated_gen
            print(f"\nExtrapolated for 81 frames:")
            print(f"  Generation time: {extrapolated_gen:.1f}s")
            print(f"  Total time: {extrapolated_total:.1f}s")
            print(f"  Target <60s: {'✓ PASSED' if extrapolated_total < 60 else '✗ FAILED'}")
        print("="*80)
    
    # Log timing summary
    log_timing_summary()

if __name__ == "__main__":
    args = parse_args()
    generate(args)
