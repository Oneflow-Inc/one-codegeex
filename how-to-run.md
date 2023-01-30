## oneflow后端启动指南
### 安装

需要Python 3.7+ / CUDA 11+ / PyTorch 1.10+ / DeepSpeed 0.6+，通过以下命令安装 ``codegeex``: 
```bash
git clone git@github.com:Oneflow-Inc/one-codegeex.git
cd one-codegeex && git checkout import_oneflow_as_torch
pip install -e . 
```

源码编译 oneflow 的 `run_codegeex` 分支。

### 模型权重

注意：目前 a100机器上有份数据，权重文件路径：`/data/home/codegeex_13b.pt`   25G

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
bash ./scripts/test_inference_oneflow.sh <GPU_ID> ./tests/test_prompt.txt
```

## PyTorch 后端启动指南




如果要启动 PyTorch 版本只需要把上面的命令改成：

```bash
# On a single GPU (with more than 27GB RAM)
bash ./scripts/test_inference.sh <GPU_ID> ./tests/test_prompt.txt
```

其它步骤比如准备模型权重和安装依赖都完全一致。

