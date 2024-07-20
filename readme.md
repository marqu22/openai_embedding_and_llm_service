辅助模型的服务构建 --yinyabo
step: 启动一个vllm 的 openai_api 风格的service
CUDA_VISIBLE_DEVICES="6" python openai_api_server.py     --model "/media/nfs2/xz_yyb/xz/xz_new/xchat_component/checkpoints_xchat_component_merged/checkpoint-2400" --port 17401 --gpu-memory-utilization 0.2    --max-log-len 20