import flask
from flask import Flask, jsonify, send_from_directory, request, render_template, make_response, send_file
import json
import time
import pandas as pd
import os
import signal
import sys
sys.path.insert(0, './super_resolution')

from option import *
#import process_ver2
import process as process
import utility as util
import common
#import torch.multiprocessing as mp
import torch.multiprocessing as mp
import time
import torch
from urllib.parse import urlparse


CSV_PATH = '/home/harim/shorts/www/analyze_outputs/video_info_for_only_swipe.csv'
DOWNLOAD_PATH = 'downloads'
VIDEO_NAME = 'MediaSegment'
INIT_NAME = 'InitializationSegment'


app = Flask(__name__, static_folder='static', template_folder='templates')

# 시퀀스 로드: sequence.json에 미리 정렬된 pid, watch_time, duration 정보가 있음
with open(os.path.join(app.static_folder, "sequence.json"), "r", encoding="utf-8") as f:
    sequence = json.load(f)
# pid -> 인덱스 매핑을 만들어 /getNeighbour에서 이용
index_map = {item["pid"]: i for i, item in enumerate(sequence)}

@app.route("/")
def index():
    return render_template("index-local.html")

# test Mock DNN
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    data = json.loads(request.form['jsondata'])
    #print('fps!!! :' + str(data['fps']))
    videodata = request.files['videofile'].read()
    quality = str(data['quality'])
    index = str(data['index'])
    
    # TODO: get resolution & framenumber
    
    vid = str(data.get('vid', 'unknown'))
    download_dir = os.path.join(DOWNLOAD_PATH, vid)
    os.makedirs(download_dir, exist_ok=True)
    print("\t uploader download dir : ",download_dir)

    file_path = os.path.join(
        download_dir, f"{quality}_{index}_{data['segmentType']}"
    )
    video_chunk_path = os.path.join(
        download_dir, f"{quality}_{index}_{VIDEO_NAME}"
    )
    init_path = os.path.join(
        download_dir, f"{quality}_None_{INIT_NAME}"
    )

    # Download a file - TODO: measure time
    newfile = open(file_path, "wb")
    newfile.write(videodata)
    newfile.close()

    print('dnn_server (input): {}'.format(
        quality + '_' + index + '_' + data['segmentType']))

    # TODO: remove 'and' - just for debugging
    if str(data['segmentType']) == 'MediaSegment' and int(index) > 0:
        try:
            output_input, output_output = mp.Pipe()
            video_info = util.videoInfo(float(data['fps']), float(data['duration']), int(data['quality']))
            video_info.vid = vid
            video_info.class_name = vid.split('/')[0] if '/' in vid else vid
            start_time = time.time()
            app.config['decode_queue'].put(
                (init_path, video_chunk_path, output_input, video_info))

            while(1):
                input = output_output.recv()

                if input[0] == 'output':
                    # change base media decode time #TODO: put into process.py encode()
                    print('output video: {}'.format(input[1]))
                    cmd = 'python set_hdr.py {} {}'.format(input[1], index)
                    print(cmd)
                    os.system(cmd)

                    cmd = 'cat {} >> {}'.format(init_path, input[1])
                    os.system(cmd)

                    #swap_file = input[1]
                    swap_file = os.path.join(os.getcwd(), input[1])


                    end_time = time.time()
                    print('{} is done'.format(input[1]))
                    print('thread [elapsed]: {}sec'.format(
                        end_time - start_time))

                    infer_idx = input[2]
                    if quality != '3':
                        infer_idx = int(infer_idx/2)

                    app.config['dnn_log'].write(str(input[1]) + ' thread [elapsed]: ' + str(end_time - start_time) + ' sec inference_idx: ' + str(infer_idx) +'\n')
                    app.config['dnn_log'].flush()
                    break
                else:
                    print('thread: Invalid income')
                    break

        except KeyboardInterrupt:
            resp = flask.Response('hello')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

        filename = str(index) + ',' + str(infer_idx)
        resp = send_file(swap_file, download_name=filename,
                         mimetype='application/octet-stream', as_attachment=True)
        response = make_response(resp)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        response.headers['Cache-Control'] = 'no-cache, no-store'
        return response
    else:
        resp = flask.Response('Init')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp


@app.route('/swap_log', methods=['GET', 'POST'])
def write_log():
    log = json.loads(request.data)
    print('\nswap_log: start-' +
          str(log['start']) + '\t' + 'time-' + str(log['time']) + '\n')
    resp = flask.Response('Init')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/dnn', methods=['GET', 'POST'])
def receive_dnn():
    print("=== /dnn CALLED ===")
    try:
        raw_json = request.form.get('jsondata', None)
        data = json.loads(raw_json)
        print("  parsed data:", data)

        if 'dnn' not in request.files:
            print("  [ERR] no 'dnn' file in request.files")
            return flask.Response("missing dnn file", status=400)

        file_name = str(data.get('file_name', 'unknown.pt'))  # "decoder.pt" or "coarse_sr.pt"
        class_name = str(data.get('class', 'unknown'))

        dnn_bytes = request.files['dnn'].read()
        print(f"  file={file_name}  class={class_name}  size={len(dnn_bytes)}B")

        download_dir = os.path.join(DOWNLOAD_PATH, class_name)
        os.makedirs(download_dir, exist_ok=True)

        save_path = os.path.join(download_dir, file_name)
        with open(save_path, "wb") as f:
            f.write(dnn_bytes)
        print("  saved to:", os.path.abspath(save_path))

        # decoder.pt + coarse_sr.pt 둘 다 도착하면 dnn_queue에 push
        decoder_path   = os.path.join(download_dir, "decoder.pt")
        coarse_sr_path = os.path.join(download_dir, "coarse_sr.pt")
        if os.path.exists(decoder_path) and os.path.exists(coarse_sr_path):
            try:
                app.config['dnn_queue'].put(('dnn_model', class_name, decoder_path, coarse_sr_path))
                print(f"  both files ready → pushed dnn_model for class={class_name}")
            except Exception as e:
                print("  [WARN] dnn_queue push failed:", repr(e))

        if 'dnn_log' in app.config:
            app.config['dnn_log'].write(f'dnn recv: {class_name}/{file_name}\n')
            app.config['dnn_log'].flush()

        resp = flask.Response('OK')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    except Exception as e:
        import traceback
        print("!!! /dnn ERROR !!!")
        traceback.print_exc()
        resp = flask.Response('error', status=500)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

@app.route('/imatrix', methods=['POST'])
def receive_imatrix():
    try:
        data = json.loads(request.form.get('jsondata', '{}'))
        vid = str(data.get('vid', 'unknown'))
        class_name = vid.split('/')[0]

        imatrix_bytes = request.files['imatrix'].read()
        download_dir = os.path.join(DOWNLOAD_PATH, class_name)
        os.makedirs(download_dir, exist_ok=True)

        imatrix_path = os.path.join(download_dir, f"imatrix_{vid.replace('/', '_')}.pt")
        with open(imatrix_path, 'wb') as f:
            f.write(imatrix_bytes)

        print(f'/imatrix recv: vid={vid} size={len(imatrix_bytes)}B → {imatrix_path}')
        app.config['dnn_queue'].put(('imatrix', vid, imatrix_path))

        resp = flask.Response('OK')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        import traceback; traceback.print_exc()
        resp = flask.Response('error', status=500)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

@app.route('/dnn_config', methods=['GET', 'POST'])
def config():
    global fps
    config_start = time.time()
    try:
        cfg = json.loads(request.data)
    except Exception:
        resp = flask.Response('bad request', status=400)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    fps = int(cfg.get('frameRate', {}).get('fps', 30))

    # Codebook: cluster id + model names
    cluster_node = cfg.get('cluster', {})
    cluster_id = cluster_node.get('id', '') if isinstance(cluster_node, dict) else ''

    vid = cfg.get('vid', '')

    print(f'[dnn_config] vid={vid} cluster={cluster_id} fps={fps}')
    print(f'[dnn_config] elapsed: {time.time() - config_start:.3f}s')

    # Codebook SR은 단일 모델(decoder+coarse_sr), inference_idx 고정 = 0
    dnn_selection = 0

    resp = flask.Response('DNN Quality select: ' + str(dnn_selection))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


# player.js (Swipe or Playback...etc...)

@app.route("/uploadPlayback")
def upload_playback():
    pid = request.args.get("vid")
    duration = request.args.get("duration")
    watch_time = request.args.get("watch_time")
    ts = time.time()
    with open("playback_log.csv", "a", encoding="utf-8") as f:
        f.write(f"{pid},{duration},{watch_time},{ts}\n")
    return jsonify({"result": 0})

with open(os.path.join(app.static_folder, "sequence.json"), "r", encoding="utf-8") as f:
    sequence = json.load(f)

@app.route("/uploadRebuffer")
def upload_rebuffer():
    pid = request.args.get("vid","")           # 어떤 영상에서
    d   = float(request.args.get("d","0") or 0) # 끊김 길이(초)
    ts  = time.time()                           # 기록 시각(초)
    with open("rebuffer_log.csv","a",encoding="utf-8") as f:
        f.write(f"{pid},{d:.3f},{ts}\n")       # 한 줄 append
    return jsonify({"result":0})


@app.route('/getScenario')
def get_scenario():
    df = pd.read_csv(CSV_PATH)
    arr = df[['pid', 'watch_time', 'duration']].to_dict('records')
    return jsonify(arr)

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    print(f"실험 시나리오 csv: {CSV_PATH}")
    mp.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    process.cudnn.benchmark = True
    signal.signal(signal.SIGINT, process.signal_handler)

    #Queue, Pipe
    decode_queue = mp.Queue()
    dnn_queue = mp.JoinableQueue()
    data_queue = mp.JoinableQueue()
    encode_queue = mp.JoinableQueue()
    output_output, output_input = mp.Pipe(duplex=False)
    test_output, test_input = mp.Pipe(duplex=False)

    #Shared tensor
    shared_tensor_list = {}
    res_list = [(270, 480), (360, 640), (540, 960), (1080, 1920)]
    for res in res_list:
        shared_tensor_list[res[0]] = []
        for _ in range(process.SHARED_QUEUE_LEN):
            shared_tensor_list[res[0]].append(torch.ByteTensor(res[1], res[0], 3).cuda().share_memory_())
            # key=width, shape=(H, W, C) portrait: ByteTensor(H=res[1], W=res[0], 3)


    #Creat processes
    decode_process = mp.Process(target=process.decode, args=(decode_queue, encode_queue, data_queue, shared_tensor_list))
    sr_process = mp.Process(target=process.super_resolution_threading, args=(encode_queue, dnn_queue, data_queue, shared_tensor_list))
    encode_process = mp.Process(target=process.encode, args=(encode_queue, shared_tensor_list))

    sr_process.start()
    decode_process.start()
    encode_process.start()

    # pretrained_path = os.path.join(opt.modelDir, 'epoch_%d.pth' % (opt.testEpoch))

    # Model upload: for pre-load DNN chunks
    """
    dnn_queue.put(('model', pretrained_path))
    time.sleep(5)

    # Run dummy jobs
    print('============DUMMY START==============')
    for dummy_process in dummy_process_list:
        dummy_process.start()
        dummy_process.join()
    print('============DUMMY END==============')
    """

    # Setup
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    dnn_log = open('./dnn_log', "w")

    # Run flask
    app.config['decode_queue'] = decode_queue
    app.config['dnn_queue'] = dnn_queue
    app.config['dnn_log'] = dnn_log
    
    app.run(host='0.0.0.0', port=8081, debug=False, use_reloader=False, threaded=True)
    
    # Join other processes
    sr_process.join()
    decode_process.join()
    encode_process.join()
