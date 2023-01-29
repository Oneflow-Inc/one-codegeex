## oneflow后端启动指南
### 安装

需要Python 3.7+ / CUDA 11+ / PyTorch 1.10+ / DeepSpeed 0.6+，通过以下命令安装 ``codegeex``: 
```bash
git clone git@github.com:THUDM/CodeGeeX.git
cd one-codegeex && git checkout import_oneflow_as_torch
pip install cpm_kernels
pip install -e . 
```

### 模型权重

通过[该链接](https://models.aminer.cn/codegeex/download/request)申请权重，您将收到一个包含临时下载链接文件```urls.txt```的邮件。推荐使用[aria2](https://aria2.github.io/)通过以下命令快速下载（请保证有足够的硬盘空间存放权重（～26GB））：
```bash
aria2c -x 16 -s 16 -j 4 --continue=true -i urls.txt 
``` 
使用以下命令合并得到完整的权重：
```bash
cat codegeex_13b.tar.gz.* > codegeex_13b.tar.gz
tar xvf codegeex_13b.tar.gz
```

### 用GPU进行推理

尝试使用CodeGeeX模型生成第一个程序吧！首先，在配置文件``configs/codegeex_13b.sh``中写明存放权重的路径。其次，将提示（可以是任意描述或代码片段）写入文件``tests/test_prompt.txt``，运行以下脚本即可开始推理（需指定GPU序号）：
```bash
# On a single GPU (with more than 27GB RAM)
bash ./scripts/test_inference.sh <GPU_ID> ./tests/test_prompt.txt

# With quantization (with more than 15GB RAM)
bash ./scripts/test_inference_quantized.sh <GPU_ID> ./tests/test_prompt.txt

# On multiple GPUs (with more than 6GB RAM, need to first convert ckpt to MP_SIZE partitions)
bash ./scripts/convert_ckpt_parallel.sh <LOAD_CKPT_PATH> <SAVE_CKPT_PATH> <MP_SIZE>
bash ./scripts/test_inference_parallel.sh <MP_SIZE> ./tests/test_prompt.txt
```


