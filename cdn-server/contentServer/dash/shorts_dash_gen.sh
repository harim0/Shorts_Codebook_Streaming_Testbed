#!/bin/bash

# Shorts DASH ladder generator
# Structure : data/{class}/{vid}.mp4  →  data/{class}/{vid}/multi_resolution.mpd
# Resolution: 270×480 portrait (9:16), bicubic 1/4 downsample from 1080×1920
# Codecs    : h.264 (libx264, 400k) + VP9 (libvpx-vp9, 400k) — both in one MPD
# Segment   : 4-second GOP (120 frames @ 30fps)

DATA_ROOT="/home/harim/Shorts_Codebook_Streaming_Testbed/cdn-server/contentServer/dash/data"

function encode_h264() {
    local input=$1
    local output=$2
    ffmpeg -y -i "${input}" \
        -c:v libx264 -b:v 400k \
        -vf "scale=270:480:flags=bicubic" \
        -preset slow \
        -x264-params "keyint=120:min-keyint=120:scenecut=0" \
        -an "${output}"
}

function encode_vp9() {
    local input=$1
    local output=$2
    ffmpeg -y -i "${input}" \
        -c:v libvpx-vp9 -b:v 400k \
        -vf "scale=270:480:flags=bicubic" \
        -g 120 -keyint_min 120 \
        -deadline good -cpu-used 2 \
        -an "${output}"
}

function make_dash() {
    local h264=$1
    local vp9=$2
    local out_dir=$3
    ffmpeg -y \
        -i "${h264}" \
        -i "${vp9}" \
        -map 0:v -c:v copy \
        -map 1:v -c:v copy \
        -f dash \
        -seg_duration 4 \
        -init_seg_name 'init_$RepresentationID$.mp4' \
        -media_seg_name 'segment_$RepresentationID$_$Number$.m4s' \
        -use_template 1 \
        -use_timeline 0 \
        "${out_dir}/multi_resolution.mpd"
}

echo "=== Shorts DASH generator (270x480 portrait, h264+vp9) ==="

for class_dir in "${DATA_ROOT}"/*/; do
    [ -d "${class_dir}" ] || continue
    class=$(basename "${class_dir}")

    for input_mp4 in "${class_dir}"*.mp4; do
        [ -f "${input_mp4}" ] || continue
        vid=$(basename "${input_mp4}" .mp4)
        out_dir="${class_dir}${vid}"

        if [ -f "${out_dir}/multi_resolution.mpd" ]; then
            echo "[SKIP] ${class}/${vid}: already done"
            continue
        fi

        echo ">>> processing: ${class}/${vid}"
        mkdir -p "${out_dir}"

        echo "  - h.264 encode (270x480 bicubic)"
        encode_h264 "${input_mp4}" "${out_dir}/h264_400k.mp4"

        echo "  - VP9 encode (270x480 bicubic)"
        encode_vp9 "${input_mp4}" "${out_dir}/vp9_400k.mp4"

        echo "  - DASH segmentation (h.264 + VP9 → single MPD)"
        make_dash "${out_dir}/h264_400k.mp4" "${out_dir}/vp9_400k.mp4" "${out_dir}"

        echo "[OK] ${class}/${vid} done."
        echo
    done
done

echo "=== All done ==="
