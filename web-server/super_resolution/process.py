#Python
import argparse, os, sys, logging, random, time, queue, signal, copy
from subprocess import Popen
import subprocess as sub
from shutil import copyfile
import numpy as np
from skimage.io import imsave
import threading
#Project
# from dataloader import *
from dataset import *
import utility as util
from option import *
from codebook_sr import CodebookSR

#PyTorch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import torch.multiprocessing as mp

import cv2 #Import Error when import cv2 before torch

TMP_DIR = 'tmp_video/'
INPUT_VIDEO_NAME = 'input.mp4'
MAX_FPS =  30
MAX_SEGMENT_LENGTH = 4
SHARED_QUEUE_LEN = MAX_FPS * MAX_SEGMENT_LENGTH #Regulate GPU memory usage (> 3 would be fine)

# ── PSNR 디버그 모드 ──────────────────────────────────────────────
# True: 청크마다 CoarseSR PSNR + E2E Final PSNR 콘솔 출력 (처리 시간 증가)
PSNR_DEBUG = True
HR_CDN_BASE = "/home/harim/Shorts_Codebook_Streaming_Testbed/cdn-server/contentServer/dash/data"
# ─────────────────────────────────────────────────────────────────

def get_resolution(quality):
    assert quality in [0, 1, 2, 3]

    # Portrait video: W=270,H=480 at LR → W=1080,H=1920 at HR (x4)
    if quality == 3:
        t_w = 1080
        t_h = 1920
    elif quality == 2:
        t_w = 540
        t_h = 960
    elif quality == 1:
        t_w = 360
        t_h = 640
    elif quality == 0:
        t_w = 270
        t_h = 480

    return (t_h, t_w)

def decode(decode_queue, encode_queue, data_queue, shared_tensor_list):
    logger = util.getLogger(opt.resultDir, 'decode.log')

    while True:
        try:
            input = decode_queue.get()
            start_time = time.time()
            #print('decode [start]: {}sec'.format(start_time))

            header_file = input[0]
            video_file = input[1]
            output_input = input[2]
            video_info = input[3]

            print(header_file)
            if not os.path.exists(header_file):
                print('decode: header does not exist')
                continue
            if not os.path.exists(video_file):
                print('decode: video does not exist')
                continue

            video_file_name, _  = os.path.splitext(os.path.basename(video_file))
            process_dir = os.path.join(TMP_DIR, '{}_{}_{}'.format(opt.contentType, video_file_name, video_info.quality))
            os.makedirs(process_dir, exist_ok=True)

            #Merge Header.m4s and X.m4s
            input_video = os.path.join(process_dir, INPUT_VIDEO_NAME)
            with open(input_video, 'wb')as outfile:
                with open(header_file, 'rb') as infile:
                    outfile.write(infile.read())

                with open(video_file, 'rb') as infile:
                    outfile.write(infile.read())

            #Call super-resolution / encoder processes to start
            current_vid = getattr(video_info, 'vid', None)
            current_class = getattr(video_info, 'class_name', None)
            
            print(f'[DECODE-START] vid={current_vid} class={current_class} quality={video_info.quality}')
            print(f'[DECODE-START] header_file={header_file}')
            print(f'[DECODE-START] video_file={video_file}')
            print(f'[DECODE-START] process_dir={process_dir}')

            data_queue.put(('process_dir', process_dir, current_vid))
            data_queue.join()

            encode_queue.put(('start', process_dir, video_info))

            t_h, t_w = get_resolution(video_info.quality)
            # targetScale: 4x SR from LR width → HR width=1080
            targetScale = int(1080 / t_w)
            targetWidth = t_w  # width-based key for shared_tensor_list
            print(f'[DECODE-CONFIG] vid={current_vid} class={current_class} '
                f'targetScale={targetScale} targetWidth={targetWidth} t_h={t_h} t_w={t_w}')
            # configure sends targetWidth so SR process uses width key
            data_queue.put(('configure', targetScale, targetWidth, current_vid, current_class))
            data_queue.join()
            print('decode [configuration]: {}sec'.format(time.time() - start_time))

            #Read frame and prepare PyTorch CUDA tensors
            vc = cv2.VideoCapture(input_video)
            frame_count = 0
            print('decode [video read prepare]: {}sec'.format(time.time() - start_time))
            while True:
                #Read frames
                rval, frame = vc.read()
                if rval == False:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # cv2.resize dsize=(width, height): portrait LR → (H=t_h, W=t_w, 3)
                frame = cv2.resize(frame, (t_w, t_h), interpolation=cv2.INTER_CUBIC)
                input_t_ = torch.from_numpy(frame).byte().cuda()

                # Store with width-based key; tensor shape (H=t_h, W=t_w, 3) portrait ✓
                shared_tensor_list[t_w][frame_count % SHARED_QUEUE_LEN].copy_(input_t_)
                data_queue.put(('frame', frame_count))
                frame_count += 1
            vc.release()

            print('decode [prepare_frames-{} frames]: {}sec'.format(frame_count, time.time() - start_time))
            data_queue.join()
            print('decode [super-resolution end]: {}sec'.format(time.time() - start_time))
            encode_queue.join() #wait for encode to be ended
            encode_queue.put(('end', output_input))
            encode_queue.join()
            print('decode [encode end] : {}sec'.format(time.time() - start_time))

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break
    print(f'[DECODE-END] vid={current_vid} frames={frame_count}')

model = CodebookSR(device='cuda')

IMATRIX_STORE = {}
IMATRIX_LOCK = threading.Lock()
ACTIVE_IMATRIX_VID = None
ACTIVE_CLASS_NAME = None
ACTIVE_MODEL_CLASS = None
LAST_CONFIGURED_VID = None
LAST_PROCESS_DIR = None

def load_dnn_chunk(dnn_queue):
    global model, ACTIVE_MODEL_CLASS
    while True:
        try:
            input = dnn_queue.get()

            if input[0] == 'dnn_model':
                # ('dnn_model', class_name, decoder_path, coarse_sr_path)
                class_name     = input[1]
                decoder_path   = input[2]
                coarse_sr_path = input[3]
                start_time = time.time()

                print(f'[DNN-QUEUE] recv dnn_model class={class_name}')
                print(f'[DNN-QUEUE] decoder={decoder_path}')
                print(f'[DNN-QUEUE] coarse_sr={coarse_sr_path}')
                print(f'[DNN-QUEUE] before load ACTIVE_MODEL_CLASS={ACTIVE_MODEL_CLASS}')

                model.load_model(decoder_path, coarse_sr_path)
                ACTIVE_MODEL_CLASS = class_name

                print('dnn_model loaded [class={}] elapsed: {:.3f}sec'.format(
                    class_name, time.time() - start_time))
                print(f'[DNN-QUEUE] after load ACTIVE_MODEL_CLASS={ACTIVE_MODEL_CLASS}')

            elif input[0] == 'imatrix':
                # ('imatrix', vid, imatrix_path)
                vid = input[1]
                imatrix_path = input[2]
                start_time = time.time()
                imatrix = torch.load(imatrix_path, map_location='cpu')

                with IMATRIX_LOCK:
                    IMATRIX_STORE[vid] = imatrix
                    cache_keys = list(IMATRIX_STORE.keys())

                print(f'[DNN-QUEUE] recv imatrix vid={vid}')
                print(f'[DNN-QUEUE] imatrix_path={imatrix_path}')
                print('imatrix cached: vid={} path={} shape={} elapsed:{:.3f}sec'.format(
                    vid, imatrix_path, tuple(imatrix.shape), time.time() - start_time))
                print(f'[DNN-QUEUE] cached vids now={cache_keys}')

            else:
                print('sr: unhandled dnn_queue message:', input[0])

            dnn_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

# def process_video_chunk(encode_queue, shared_tensor_list, data_queue):
#     global model, ACTIVE_IMATRIX_VID, ACTIVE_CLASS_NAME

#     targetHeight = None
#     process_dir = None
#     current_vid = None
#     current_class = None
#     inference_time_list = []

#     while True:
#         try:
#             input = data_queue.get()

#             if input[0] == 'configure':
#                 targetHeight = input[2]
#                 current_vid = input[3] if len(input) > 3 else None
#                 current_class = input[4] if len(input) > 4 else None
#                 inference_time_list = []

#                 # Codebook SR은 항상 4x (270→1080), index 고정
#                 encode_queue.put(('index', 0))

#                 print(f'[SR-CONFIG] vid={current_vid} class={current_class} targetHeight={targetHeight}')

#             elif input[0] == 'process_dir':
#                 process_dir = input[1]
#                 if len(input) > 2:
#                     current_vid = input[2]

#             elif input[0] == 'frame':
#                 start_time = time.time()
#                 frame_count = input[1]

#                 curr_idx = frame_count % SHARED_QUEUE_LEN
#                 prev_idx = (frame_count - 1) % SHARED_QUEUE_LEN if frame_count > 0 else curr_idx

#                 lr_prev = shared_tensor_list[targetHeight][prev_idx]
#                 lr_curr = shared_tensor_list[targetHeight][curr_idx]

#                 # 현재 vid에 맞는 imatrix 적용
#                 selected_imatrix = None
#                 if current_vid is not None:
#                     with IMATRIX_LOCK:
#                         selected_imatrix = IMATRIX_STORE.get(current_vid, None)

#                 if selected_imatrix is not None:
#                     if ACTIVE_IMATRIX_VID != current_vid:
#                         model.set_imatrix(selected_imatrix)
#                         ACTIVE_IMATRIX_VID = current_vid
#                         print(f'[SR-IMATRIX] switched to vid={current_vid} shape={tuple(selected_imatrix.shape)}')

#                     output_ = model.infer_with_imatrix(lr_prev, lr_curr, frame_count)
#                 else:
#                     if ACTIVE_IMATRIX_VID is not None:
#                         # 현재 비디오에 해당하는 imatrix가 없으면 global 상태 제거
#                         model.set_imatrix(None)
#                         ACTIVE_IMATRIX_VID = None
#                         print(f'[SR-IMATRIX] cleared for vid={current_vid}')

#                     output_ = model.infer_tensor(lr_prev, lr_curr, lr_curr)

#                 shared_tensor_list[1080][curr_idx].copy_(output_)
#                 torch.cuda.synchronize()

#                 encode_queue.put(('frame', curr_idx))
#                 end_time = time.time()

#                 if opt.enable_debug:
#                     output_np = output_.float().cpu().numpy().astype(np.uint8)
#                     imsave(os.path.join('{}/sr_{}p_{}.png'.format(
#                         process_dir, targetHeight, curr_idx)), output_np)

#                 inference_time_list.append(end_time - start_time)

#                 if frame_count == 95:
#                     print('process [codebook][vid={}] total-{}frames: {:.3f}sec'.format(
#                         current_vid, len(inference_time_list), np.sum(inference_time_list)))
#             else:
#                 print('sr: Invalid input')

#             data_queue.task_done()

#         except (KeyboardInterrupt, SystemExit):
#             print('exiting...')
#             break

def _load_hr_frames(vid):
    """vid(예: 'Animal/SDR_Animal_3k7l') → HR MP4 로드 후 RGB uint8 프레임 리스트 반환."""
    hr_path = os.path.join(HR_CDN_BASE, vid + '.mp4')
    if not os.path.exists(hr_path):
        print(f'[PSNR] HR file not found: {hr_path}')
        return None
    cap = cv2.VideoCapture(hr_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f'[PSNR] loaded HR {vid}: {len(frames)} frames from {hr_path}')
    return frames


def _psnr(pred_hwc, ref_hwc):
    """두 HWC uint8 numpy 배열 또는 CUDA tensor 간 PSNR (dB) 계산."""
    if isinstance(pred_hwc, torch.Tensor):
        pred_hwc = pred_hwc.cpu().numpy()
    if isinstance(ref_hwc, torch.Tensor):
        ref_hwc = ref_hwc.cpu().numpy()
    mse = np.mean((pred_hwc.astype(np.float32) - ref_hwc.astype(np.float32)) ** 2)
    return 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')


def process_video_chunk(encode_queue, shared_tensor_list, data_queue):
    global model, ACTIVE_IMATRIX_VID, ACTIVE_CLASS_NAME
    global ACTIVE_MODEL_CLASS, LAST_CONFIGURED_VID, LAST_PROCESS_DIR

    targetWidth = None  # LR width: 270 for quality=0 (portrait)
    process_dir = None
    current_vid = None
    current_class = None
    inference_time_list = []

    # PSNR 디버그용
    _hr_frames = None          # 현재 vid의 HR 프레임 리스트
    _hr_vid_loaded = None      # 캐시 키
    _psnr_coarse_list = []
    _psnr_final_list  = []

    while True:
        try:
            input = data_queue.get()
            print("[process_video_chunk] input : ", input)

            if input[0] == 'configure':
                # 이전 vid PSNR 요약 출력
                if PSNR_DEBUG and _psnr_coarse_list:
                    print(f'[PSNR] vid={_hr_vid_loaded} frames={len(_psnr_coarse_list)} '
                          f'CoarseSR={np.mean(_psnr_coarse_list):.2f}dB '
                          f'E2E-Final={np.mean(_psnr_final_list):.2f}dB')

                targetWidth = input[2]  # width-based key (270 for LR, 1080 for HR)
                current_vid = input[3] if len(input) > 3 else None
                current_class = input[4] if len(input) > 4 else None
                LAST_CONFIGURED_VID = current_vid
                ACTIVE_CLASS_NAME = current_class
                inference_time_list = []
                _psnr_coarse_list = []
                _psnr_final_list  = []

                # PSNR 디버그: 새 vid HR 로드
                if PSNR_DEBUG and current_vid and current_vid != _hr_vid_loaded:
                    _hr_frames = _load_hr_frames(current_vid)
                    _hr_vid_loaded = current_vid if _hr_frames else None

                encode_queue.put(('index', 0))

                print(f'[SR-CONFIG] vid={current_vid} class={current_class} '
                      f'targetWidth={targetWidth} ACTIVE_MODEL_CLASS={ACTIVE_MODEL_CLASS}')

            elif input[0] == 'process_dir':
                process_dir = input[1]
                LAST_PROCESS_DIR = process_dir
                if len(input) > 2:
                    current_vid = input[2]
                print(f'[SR-PROCESS-DIR] vid={current_vid} process_dir={process_dir}')

            elif input[0] == 'frame':
                start_time = time.time()
                frame_count = input[1]

                curr_idx = frame_count % SHARED_QUEUE_LEN
                prev_idx = (frame_count - 1) % SHARED_QUEUE_LEN if frame_count > 0 else curr_idx

                # shared_tensor keyed by width: (270 → portrait H=480,W=270,C=3)
                lr_prev = shared_tensor_list[targetWidth][prev_idx]
                lr_curr = shared_tensor_list[targetWidth][curr_idx]

                selected_imatrix = None
                if current_vid is not None:
                    with IMATRIX_LOCK:
                        selected_imatrix = IMATRIX_STORE.get(current_vid, None)

                if frame_count in (0, 1, 2, 30, 60, 90):
                    print(f'[SR-FRAME] vid={current_vid} class={current_class} frame={frame_count} '
                          f'ACTIVE_MODEL_CLASS={ACTIVE_MODEL_CLASS} '
                          f'ACTIVE_IMATRIX_VID={ACTIVE_IMATRIX_VID} '
                          f'imatrix_hit={selected_imatrix is not None}')

                if selected_imatrix is not None:
                    if ACTIVE_IMATRIX_VID != current_vid:
                        model.set_imatrix(selected_imatrix)
                        ACTIVE_IMATRIX_VID = current_vid
                        print(f'[SR-IMATRIX] switched to vid={current_vid} shape={tuple(selected_imatrix.shape)} '
                              f'ACTIVE_MODEL_CLASS={ACTIVE_MODEL_CLASS}')

                    if PSNR_DEBUG and _hr_frames is not None:
                        output_, x_coarse_hwc = model.infer_with_imatrix(
                            lr_prev, lr_curr, frame_count, return_coarse=True)
                        hr_idx = frame_count % len(_hr_frames)
                        hr_frame = _hr_frames[hr_idx]
                        # HR 크기 맞추기 (모델 출력과 다를 경우)
                        h, w = output_.shape[:2] if isinstance(output_, torch.Tensor) else output_.shape[:2]
                        if hr_frame.shape[0] != h or hr_frame.shape[1] != w:
                            hr_frame = cv2.resize(hr_frame, (w, h), interpolation=cv2.INTER_AREA)
                        _psnr_coarse_list.append(_psnr(x_coarse_hwc, hr_frame))
                        _psnr_final_list.append(_psnr(output_, hr_frame))
                    else:
                        output_ = model.infer_with_imatrix(lr_prev, lr_curr, frame_count)

                    if frame_count in (0, 1, 2):
                        print(f'[SR-INFER] mode=with_imatrix vid={current_vid} frame={frame_count}')
                else:
                    if ACTIVE_IMATRIX_VID is not None:
                        model.set_imatrix(None)
                        ACTIVE_IMATRIX_VID = None
                        print(f'[SR-IMATRIX] cleared for vid={current_vid}')

                    output_ = model.infer_tensor(lr_prev, lr_curr, lr_curr)

                    if frame_count in (0, 1, 2):
                        print(f'[SR-INFER] mode=fallback_tensor vid={current_vid} frame={frame_count}')

                shared_tensor_list[1080][curr_idx].copy_(output_)
                torch.cuda.synchronize()

                encode_queue.put(('frame', curr_idx))
                end_time = time.time()

                inference_time_list.append(end_time - start_time)

                if frame_count == 95:
                    print('process [codebook][vid={}] total-{}frames: {:.3f}sec'.format(
                        current_vid, len(inference_time_list), np.sum(inference_time_list)))
                    print(f'[SR-SUMMARY] vid={current_vid} class={current_class} '
                          f'ACTIVE_MODEL_CLASS={ACTIVE_MODEL_CLASS} '
                          f'ACTIVE_IMATRIX_VID={ACTIVE_IMATRIX_VID} '
                          f'process_dir={process_dir}')
                    if PSNR_DEBUG and _psnr_coarse_list:
                        print(f'[PSNR] vid={current_vid} chunk_avg(0~95) '
                              f'CoarseSR={np.mean(_psnr_coarse_list):.2f}dB '
                              f'E2E-Final={np.mean(_psnr_final_list):.2f}dB')
            else:
                print('sr: Invalid input')

            data_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

def super_resolution_threading(encode_queue, dnn_queue, data_queue, shared_tensor_list):
    dnn_load_thread = threading.Thread(target=load_dnn_chunk, args=(dnn_queue,))
    video_process_thread = threading.Thread(target=process_video_chunk, args=(encode_queue, shared_tensor_list, data_queue))
    dnn_load_thread.start()
    video_process_thread.start()
    dnn_load_thread.join()
    video_process_thread.join()

#TODO (minor) : 1. remove all images
def encode(encode_queue, shared_tensor_list):
    pipe = None
    process_dir = None
    infer_idx = None

    while(1):
        try:
            input = encode_queue.get()

            if input[0] == 'start':
                encode_start_time = time.time()
                #print('encode [start]: {}sec'.format(encode_start_time))

                process_dir = input[1]
                video_info = input[2]

                fps = video_info.fps
                duration = video_info.duration
                total_frames = duration * fps

                print('encode [after video info]: {}sec'.format(time.time() - encode_start_time))

                command = [ '/usr/bin/ffmpeg',
                            '-r', str(fps), # frames per second
                            '-y',
                            '-loglevel', 'error',
                            '-f', 'rawvideo',
                            '-vcodec','rawvideo',
                            '-s', '1080x1920', # portrait: W=1080, H=1920
                            #'-s', '1280x720', # size of one frame
                            '-pix_fmt', 'rgb24',
                            '-i', '-', # The imput comes from a pipe
                            #'-s', '1920x1080', # size of one frame
                            '-vcodec', 'libx264',
                            #'crf', '0',
                            '-preset', 'ultrafast',
                            '-movflags', 'empty_moov+omit_tfhd_offset+frag_keyframe+default_base_moof',
                            '-pix_fmt', 'yuv420p',
                            #'-an', # Tells FFMPEG not to expect any audio
                            '{}'.format(os.path.join(process_dir, 'output.mp4'))]

                pipe = sub.Popen(command, stdin=sub.PIPE, stderr=sub.PIPE)
                end_time_ = time.time()
                print('encode [start]: {}sec'.format(end_time_ - encode_start_time))

            elif input[0] == 'frame':
                #start_time_ = time.time()
                idx = input[1]
                img = shared_tensor_list[1080][idx].cpu().numpy()

                if img is None:
                    print(idx)

                pipe.stdin.write(img.tobytes())
                pipe.stdin.flush()
                #end_time_ = time.time()
                #print('encode [frame]: {}sec'.format(end_time_ - start_time_))

            elif input[0] == 'end':
                start_time_ = time.time()
                pipe.stdin.flush()
                pipe.stdin.close()
                pipe = None
                output_input = input[1]
                #infer_idx = input[2]
                #infer_idx = -1 #TODO

                #print('encode [end] : {}sec'.format(end_time))
                encode_end_time = time.time()
                print('encode [end]: {}sec'.format(encode_end_time - start_time_))
                print('encode [elapsed] / index [{}]: {}sec'.format(infer_idx, encode_end_time - encode_start_time))

                output_input.send(('output', os.path.join(process_dir, 'output.mp4'), infer_idx))
                process_dir = None

            elif input[0] == 'dummy':
                print('dummy run')
                img = input[1].cpu()
                continue

            elif input[0] == 'index':
                infer_idx = input[1]

            else:
                print('encode: Invalid input')
                continue

            encode_queue.task_done()

        except (KeyboardInterrupt, SystemExit):
            print('exiting...')
            break

#Test for off-line : request [resolution]/[index].mp4
def request(decode_queue, resolution, index):
    res2bitrate = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
    #print('overall [start]: {}sec'.format(start_time))

    start_time = time.time()
    video_info = util.videoInfo(24.0, 4.0, res2quality[resolution])
    output_output, output_input = mp.Pipe(duplex=False)
    decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))
    #decode_queue.put((os.path.join(video_dir, 'output_{}k_dashinit.mp4'.format(res2bitrate[resolution])), os.path.join(video_dir, 'output_{}k_dash{}.m4s'.format(res2bitrate[resolution], index)), output_input, video_info)) #temporary for old encoding formats

    while(1):
        input = output_output.recv()
        if input[0] == 'output':
            end_time = time.time()
            #print('overall [end] : {}sec'.format(end_time))
            print('overall [elapsed], resolution [{}p] : {}sec'.format(resolution, end_time - start_time))
            break
        else:
            print('request: Invalid input')
            break

def test():
    tensor = torch.FloatTensor(96, 3, 1920, 1080).random_(0,1)
    tensor = tensor.cuda()

    start_time = time.time()
    tensor_cpu = tensor.cpu()
    end_time = time.time()

    print('gpu2cpu: {}sec'.format(end_time - start_time))

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

#Test for off-line : request [resolution]/[index].mp4
def test_figure16(clock, index=1):
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #Creat processes
    decode_process = mp.Process(target=decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=encode, args=(encode_queue, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    #Configuration
    res2bitrate = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    #resolution_list = [240, 360, 480, 720]
    resolution_list = [240]
    segment_fps = 30
    segment_size = 4

    decode_process.start()
    sr_process.start()
    encode_process.start()

    print('=====dummy start=====')
    for resolution in resolution_list:
        video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
        video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
        output_output, output_input = mp.Pipe(duplex=False)
        decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))

        while(1):
            input = output_output.recv()
            if input[0] == 'output':
                break
            else:
                print('request: Invalid input')
                break
    print('=====dummy_end=====')

    #Iterate over xx times - report min/average/max
    elapsed_time_list = {}
    fps_list = {}

    output_list = [0,1,2,3,4]
    for output in output_list:
        elapsed_time_list[output] = {}
        fps_list[output] = {}
        for resolution in resolution_list:
            elapsed_time_list[output][resolution] = []
            fps_list[output][resolution] = []

    #Set inference index
    for _ in range(opt.runtimeNum):
        for output in output_list:
            dnn_queue.put(('test_figure16',output,))
            dnn_queue.join()
            for resolution in resolution_list:
                video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
                video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
                output_output, output_input = mp.Pipe(duplex=False)
                start_time = time.time()
                decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))
                while(1):
                    input = output_output.recv()
                    if input[0] == 'output':
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        fps = segment_fps * segment_size / (end_time - start_time)
                        print('overall [elapsed], resolution [{}p] : {} second, {} fps'.format(resolution, elapsed_time, fps))
                        elapsed_time_list[output][resolution].append(elapsed_time)
                        fps_list[output][resolution].append(fps)
                        break
                    else:
                        print('request: Invalid input')
                        break

    #Log
    os.makedirs('dnn_runtime_{}'.format(clock), exist_ok=True)
    runtimeLogger = util.getLogger('dnn_runtime_{}'.format(clock), opt.quality)

    #Print statistics
    for output in output_list:
        for resolution in resolution_list:
            print('[output: {}][{}p]: minmum {} fps, average {} fps, maximum {} fps'.format(output, resolution, np.min(fps_list[output][resolution]), np.average(fps_list[output][resolution]), np.max(fps_list[output][resolution])))
            log_str = "\t".join(map(str, fps_list[output][resolution]))
            log_str += "\t{}".format(np.average(fps_list[output][resolution]))
            runtimeLogger.info(log_str)

    sr_process.terminate()
    decode_process.terminate()
    encode_process.terminate()

#Test for off-line : request [resolution]/[index].mp4
def test_runtime(index=1):
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #Creat processes
    decode_process = mp.Process(target=decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=encode, args=(encode_queue, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    #Configuration
    res2bitrate = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    #resolution_list = [240, 360, 480, 720]
    resolution_list = [240, 360, 480, 720]
    segment_fps = 30
    segment_size = 4

    decode_process.start()
    sr_process.start()
    encode_process.start()

    #Set inference index
    dnn_queue.put(('test_runtime',))
    time.sleep(1)

    print('=====dummy start=====')
    for resolution in resolution_list:
        video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
        video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
        output_output, output_input = mp.Pipe(duplex=False)
        decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))

        while(1):
            input = output_output.recv()
            if input[0] == 'output':
                break
            else:
                print('request: Invalid input')
                break
    print('=====dummy_end=====')

    #Iterate over xx times - report min/average/max
    elapsed_time_list = {}
    fps_list = {}

    for resolution in resolution_list:
        elapsed_time_list[resolution] = []
        fps_list[resolution] = []

    for _ in range(opt.runtimeNum):
        for resolution in resolution_list:
            video_dir = os.path.join(opt.videoDir, '{}p'.format(resolution))
            video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
            output_output, output_input = mp.Pipe(duplex=False)
            start_time = time.time()
            decode_queue.put((os.path.join(video_dir, 'segment_init.mp4'), os.path.join(video_dir, 'segment_{}.m4s'.format(index)), output_input, video_info))
            while(1):
                input = output_output.recv()
                if input[0] == 'output':
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    fps = segment_fps * segment_size / (end_time - start_time)
                    print('overall [elapsed], resolution [{}p] : {} second, {} fps'.format(resolution, elapsed_time, fps))
                    elapsed_time_list[resolution].append(elapsed_time)
                    fps_list[resolution].append(fps)
                    break
                else:
                    print('request: Invalid input')
                    break

    #Log
    os.makedirs('dnn_runtime', exist_ok=True)
    runtimeLogger = util.getLogger('dnn_runtime', opt.quality)

    #Print statistics
    for resolution in resolution_list:
        print('[{}p]: minmum {} fps, average {} fps, maximum {} fps'.format(resolution, np.min(fps_list[resolution]), np.average(fps_list[resolution]), np.max(fps_list[resolution])))
        log_str = "\t".join(map(str, fps_list[resolution]))
        log_str += "\t{}".format(np.average(fps_list[resolution]))
        runtimeLogger.info(log_str)

    sr_process.terminate()
    decode_process.terminate()
    encode_process.terminate()

def test_multi_resolution():
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #Creat processes
    decode_process = mp.Process(target=decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=encode, args=(encode_queue, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    #Configuration
    res2bitrate = {240: 400, 360: 800, 480: 1200, 720: 2400, 1080: 4800}
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}
    #resolution_list = [240, 360, 480, 720]
    resolution_list = [240, 360, 480, 720]
    segment_fps = 30
    segment_size = 4
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    request_process_list = []

    print ('test start')
    sr_process.start()
    decode_process.start()
    encode_process.start()

    pretrained_path = os.path.join(opt.modelDir, 'epoch_%d.pth' % (opt.testEpoch))
    dnn_queue.put(('model', pretrained_path))
    dnn_queue.join()

    dnn_idx = 4
    chunk_idx = [6, 7, 8, 9, 10]
    for _ in range(1):
        dnn_queue.put(('test_figure16',dnn_idx,)) #output = 0,1,2,3,4 - 4 is a full model
        dnn_queue.join() # wait for done
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, chunk_idx[dnn_idx])))
        dnn_idx += 1
        #request_process_list.append(mp.Process(target=request, args=(decode_queue, 360, 1)))
        #request_process_list.append(mp.Process(target=request, args=(decode_queue, 480, 1)))
        #request_process_list.append(mp.Process(target=request, args=(decode_queue, 720, 1)))



    print('============INFERENCE START==============')
    for request_process in request_process_list:
        request_process.start()
        request_process.join()
    print('============INFERENCE END==============')

    sr_process.terminate()
    decode_process.terminate()
    encode_process.terminate()

#TODO: partial model load
def test_scalable_dnn():
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.Queue()
    process_output, process_input= mp.Pipe(duplex=False)
    encode_output, encode_input = mp.Pipe(duplex=False)
    output_output, output_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = []
    for _ in range(SHARED_QUEUE_LEN):
        shared_tensor_list.append(torch.ByteTensor(1080, 1920, 3).cuda().share_memory_())
        #shared_tensor_list.append(torch.ByteTensor(720, 1280, 3).cuda().share_memory_())

    #Lock
    mp_lock_list = []
    for _ in range(SHARED_QUEUE_LEN):
        mp_lock_list.append(mp.Lock())

    decode_process = mp.Process(target=decode, args=(decode_queue, process_input))
    sr_process = mp.Process(target=super_resolution_threading, args=(process_output, encode_input, dnn_queue, mp_lock_list, shared_tensor_list))
    #sr_process = mp.Process(target=super_resolution, args=(process_output, encode_input))
    encode_process = mp.Process(target=encode, args=(encode_output, mp_lock_list, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    for _ in range(1):
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 1)))

    for _ in range(1):
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 1)))
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 2)))
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 3)))
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 4)))
        request_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 5)))

    sr_process.start()
    decode_process.start()
    encode_process.start()

    print('============DUMMY START==============')
    for dummy_process in dummy_process_list:
        dummy_process.start()
        dummy_process.join()
    print('============DUMMY END==============')

    print('============INFERENCE START==============')
    count = 0
    for request_process in request_process_list:
        pretrained_path = os.path.join(opt.modelDir, 'DNN_chunk_{}.pth'.format(count))
        dnn_queue.put(('dnn_chunk', pretrained_path, count))
        time.sleep(0.5)
        request_process.start()
        request_process.join()
        count += 1
    print('============INFERENCE END==============')

#Assumption: 1) encode + decode overhead: prefixed constant (x sec) 2) resolution of which has the most overhead is known 3) 'feature' represets the number of body channels (~2x of pre-process and post-process)
#Test with low to scale4
#TODO: find 2) - currently using 1 / pass fps / duration

def run_dummy():
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    decode_queue = mp.Queue()
    process_output, process_input= mp.Pipe(duplex=False)
    encode_output, encode_input = mp.Pipe(duplex=False)
    output_output, output_input = mp.Pipe(duplex=False)

    decode_process = mp.Process(target=decode, args=(decode_queue, process_input))
    sr_process = mp.Process(target=super_resolution, args=(process_output, encode_input))
    encode_process = mp.Process(target=encode, args=(encode_output, 1))

    dummy_process_list = []
    request_process_list = []

    for _ in range(1):
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 360, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 480, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 720, 1)))

    sr_process.start()
    decode_process.start()
    encode_process.start()

    print('============DUMMY START==============')
    for dummy_process in dummy_process_list:
        dummy_process.start()
        dummy_process.join()
    print('============DUMMY END==============')

    decode_process.terminate()
    sr_process.terminate()
    encode_process.terminate()

def test_mock_DNNs():
    cudnn.benchmark = True
    signal.signal(signal.SIGINT, signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)
    test_output, test_input= mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())
            #shared_tensor_list[res[0]].append(torch.ByteTensor(res[0], res[1], 3).cuda().share_memory_())

    #Creat processes
    decode_process = mp.Process(target=decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=encode, args=(encode_queue, shared_tensor_list))

    dummy_process_list = []
    request_process_list = []

    dummy_process_list = []
    request_process_list = []

    for _ in range(1):
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 240, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 360, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 480, 1)))
        dummy_process_list.append(mp.Process(target=request, args=(decode_queue, 720, 1)))

    sr_process.start()
    decode_process.start()
    encode_process.start()

    """
    print('============DUMMY START==============')
    for dummy_process in dummy_process_list:
        dummy_process.start()
        dummy_process.join()
    print('============DUMMY END==============')
    """

    print('============TEST START==============')
    DNN_list = [(18, 9), (18, 21), (18, 32), (18, 48)]
    fps = 30
    duration = 4
    dnn_queue.put(('test_dnn', DNN_list, fps, duration, test_input))

    input = test_output.recv()
    print('Selected DNN index is {}'.format(input[0]))
    print('============TEST END==============')

    decode_process.terminate()
    sr_process.terminate()
    encode_process.terminate()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    if opt.processMode == 'multi':
        test_multi_resolution()
    elif opt.processMode == 'scalable':
        test_scalable_dnn()
    elif opt.processMode == 'mock':
        test_mock_DNNs()
    elif opt.processMode == 'runtime':
        test_runtime()
    elif opt.processMode == 'figure16':
        CLOCK_INFO = {}
        #TITAN_XP_INFO = [1404, 1303, 1202, 1101, 999, 898, 797, 696, 506]
        #TITAN_XP_INFO = [1404, 1303, 1202, 1101, 999, 898, 797, 696]
        TITAN_XP_INFO = [949]
        TITAN_XP_INFO.reverse()
        CLOCK_INFO['titanxp'] = TITAN_XP_INFO

        for clock in CLOCK_INFO['titanxp']:
            os.system('echo ina8024 | sudo -S nvidia-smi -i 0 --applications-clocks=5705,{}'.format(clock))
            test_figure16(clock)
        os.system('echo ina8024 nvidia-smi -i 0 --reset-applications-clocks')
