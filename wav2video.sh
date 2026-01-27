PYTHON_SCRIPT="./wav2video.py"
SERVER_URL="http://127.0.0.1:8084/send_json"
OVERWRITE=false
# 定义批处理任务列表
# 格式: "json_path|wave_path|video_path|default_wave_path"
# 注意: default_wave_path是可选地，若为空或未提供，则不传递--default_wave_path参数
tasks=(
  ""
  ""
  ""
)
for task in "${tasks[@]}"; do






