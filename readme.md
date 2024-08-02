Openai-compatible api interface based on fastapi
    1. Implement the LLM interface based on vllm
    2. Manually constructed OpenAI-style embedding interface

step: 启动一个vllm 的 openai_api 风格的service
node1: vllm 环境 安装: 
```bash
# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.5.0
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```
# Install vLLM with CUDA 11.8.
step:启动服务
### llm service
```
CUDA_VISIBLE_DEVICES="2" python openai_api_llm_server.py     --model "/media/nfs2/glm-4-9b-chat" --port 17401 --gpu-memory-utilization 0.2    --max-log-len 20 
```
### embedding service
```
python openai_api_embed_server.py  --port 17402 --gpu 2
```
# bash
```
curl localhost:17402/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer DEEPSEEK_API_KEY" \
  -d '{
        "model": "/media/nfs2/glm-4-9b-chat",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "stream": false
      }'
```