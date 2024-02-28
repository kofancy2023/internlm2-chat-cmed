
# internlm2-chat-cmed

以InternLM2-chat-7为基座模型，以常用中药为数据集，微调的大模型。中医聊天小助手。

# 重新训练

## 1. 环境准备

### 1.1 创建并激活虚拟环境

```python
bash 

# 创建虚拟环境
conda create --name cmed2 python=3.10 -y

# 查看虚拟环境
conda env list

# 激活虚拟环境
conda activate cmed2 

```

### 1.2 安装依赖

```python
# ----安装依赖
# 升级pip
python -m pip install --upgrade pip

pip install modelscope
pip install transformers 
pip install streamlit
pip install sentencepiece
pip install accelerate

```

### 1.3 下载代码

-   下载本项目的代码

    ```bash
    # 创建项目目录
    mkdir -p /root/cmed2/code && cd /root/cmed2/code
    
    # 从github上拉取本项目代码
    git clone https://github.com/kofancy2023/internlm2-chat-cmed.git
    cd internlm2-chat-cmed
    ```

-   安装最新版xtuner
    ```bash
    cd /root/cmed2/code
    git clone https://gitee.com/Internlm/xtuner

    # 进入源码目录
    cd xtuner

    # 从源码安装 XTuner
    pip install -e .[all]

    ```

## 2. 训练数据准备

### 2.1 准备数据集

> 通常先收集到excel中，然后再转成json格式。

> 注：可以考虑用python处理数据，产生相应的数据集。

```bash
# 创建data文件夹用于存放用于训练的数据集
mkdir -p /root/cmed2/data && cd /root/cmed2/data

# 将提前准备好的微调数据集cmed2.json放入data文件夹
cp -r  /root/cmed2/code/internlm2-chat-cmed/model/data/cmed2.json  /root/cmed2/data


```

-   json格式参考：
    ```bash
    [
        {
            "conversation": [
                {
                    "input": "请介绍一下你自己",
                    "output": "我是XXX的小助手，内在是上海AI实验室书生·浦语的7B大模型哦"
                }
            ]
        },
        {
            "conversation": [
                {
                    "input": "请做一下自我介绍",
                    "output": "我是XXX的小助手，内在是上海AI实验室书生·浦语的7B大模型哦"
                }
            ]
        }
    ]
    ```

## 3. 大语言模型准备（基座模型）

```bash
# 创建model文件夹用于存放模型
mkdir -p /root/model/Shanghai_AI_Laboratory
```

-   法一：----用InternStudio 平台默认提供的----
    ```bash
    cp -r /root/share/model_repos/internlm2-chat-7b /root/model/Shanghai_AI_Laboratory
    ```
-   法二：----用modelscope 下载----
    ```bash
    # 安装一下拉取模型文件要用的库
    pip install modelscope

    # 从 modelscope 下载模型文件internlm2-chat-7b
    cd /root/model/Shanghai_AI_Laboratory

    # linux环境下安装git lfs。如果windows下安装，参考这个网址：https://blog.csdn.net/weixin_43213884/article/details/123739511
    apt install git git-lfs -y 
    git lfs install

    # 下载模型
    git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm2-chat-7b.git

    ```
-   **法三：【推荐】用python代码来下载模型：download\_internlm2-chat-7b.py**
    > 可视化更好
    ```bash
    # pip install modelscope
    import torch
    from modelscope import snapshot_download, AutoModel, AutoTokenizer
    import os

    # cache_dir: 模型下载的缓存目录
    model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='/root/model/Shanghai_AI_Laboratory')
    ```

## 4. 配置准备

-   XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：
    ```bash
    # 列出所有内置配置
    xtuner list-cfg

    #创建用于存放配置的文件夹config并进入
    mkdir /root/cmed2/config && cd /root/cmed2/config

    ```
-   **拷贝一个配置文件到当前目录：xtuner copy-cfg **$ {CONFIG_NAME}  $**{SAVE_PATH} ，在本例中：（注意最后有个英文句号，代表复制到当前路径）**
    ```bash
    xtuner copy-cfg  internlm2_chat_7b_qlora_oasst1_e3  .

    # xtuner copy-cfg internlm2_7b_qlora_json_e3 /root/cmed/config

    # 修改配置文件名
    mv internlm2_chat_7b_qlora_oasst1_e3_copy.py  internlm2_chat_7b_qlora_cmed2_e3.py
    ```
-   修改拷贝后的文件internlm2_chat_7b_qlora_cmed2_e3.py，修改下述位置：
    ```bash
    # PART 1 中
    # 预训练模型存放的位置
     pretrained_model_name_or_path = '/root/model/Shanghai_AI_Laboratory/internlm2-chat-7b' 

    # 微调数据存放的位置
     data_path = '/root/cmed2/data/cmed2.json' 

    # 训练中最大的文本长度
    max_length = 2048 #512

    # 每一批训练样本的大小
    batch_size = 2

    # 最大训练轮数
    max_epochs = 3

    # 验证的频率
    evaluation_freq = 90

    # 用于评估输出内容的问题（用于评估的问题尽量与数据集的question保持一致）
    evaluation_inputs = [ '桂枝的用法', '桂枝的配伍', '甘草有什么功效','甘草的炮制方法有哪些','甘草的性味归经' ]


    # PART 3 中
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path))
    dataset_map_fn=None
    ```

## 5. 微调启动

-   法一：【推荐】在微调前，先安装tmux，再在tmux中进行微调（xtuner train）
    -   安装tmux，并创建session
        ```bash
        # 安装tmux
        apt install tmux -y

        # 创建tmux的session
        tmux new -s finetune

        # 从bash终端回到tmux的finetune终端，再进行训练
        # tmux attach -t finetune

        ```
    -   在tmux中进行微调（xtuner train）
        ```bash
        # 删除work_dirs文件夹
        rm -rf /root/cmed2/config/work_dirs

         cd /root/cmed2/config 

        # 修改配置文件的名称

        # 加速，减少一半的训练时间：--deepspeed deepspeed_zero2
        xtuner train /root/cmed2/config/internlm2_chat_7b_qlora_cmed2_e3.py --deepspeed deepspeed_zero2
        ```
   
-   法二：直接用xtuner train命令启动训练【太长时间会断开】
    ```bash
    # 删除work_dirs文件夹
    rm -rf /root/cmed2/config/work_dirs

    cd /root/cmed2/config

    # 加速，减少一半的训练时间：--deepspeed deepspeed_zero2
    xtuner train /root/cmed2/config/internlm2_chat_7b_qlora_cmed2_e3.py --deepspeed deepspeed_zero2

    ```
-   会在训练完成后，输出用于验证的Sample output


-   **小技巧（通过tmux微调）：**
    -   因为我们的训练基本上都会跑很长的时间，一般都是1、2个小时起步。如果一直窗口挂着，可能会因为一些问题，连接会中断，而一旦中断，远程电脑上的微调工作也被中断了。相当于按下了CTRL+C。
    -   所以要借助一个工具tmux：通过tmux，**我们即使中断了SSH连接，但微调还可以继续。**
        ```bash
        # 删除文件夹：work_dirs
        rm -rf /root/cmed2/config/work_dirs

        # 在root账号下，更新apt
        apt update -y

        # 安装tmux
        apt install tmux -y

        # 1、创建tmux的session
        tmux new -s finetune

        # 2、从tmux终端回到bash终端
        CTRL+B，双手离开键盘
        再按D

        # 3、从bash终端回到tmux的finetune终端，再进行训练
        tmux attach -t finetune

        # ------以下在tmux终端中跑的东西，可以让它一直在后台跑，即使我们SSH连接关了，它的工作还能在tmux上继续------
        # 单卡
        ## 用刚才改好的config文件训练
        xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2

        # 从tmux终端回到bash终端
        CTRL+B, 双手离开键盘
        再按D

        # 关闭本机电脑，几小时训练结束后再来看
        ```

## 6. 微调结束后，进行模型参数转换/合并

### 6.1 模型转换：将训练后的pth格式参数转Hugging Face格式

```bash
# 创建用于存放Hugging Face格式参数的hf文件夹
mkdir /root/cmed2/config/work_dirs/hf

export MKL_SERVICE_FORCE_INTEL=1

# 配置文件存放的位置
export CONFIG_NAME_OR_PATH=/root/cmed2/config/ internlm2_chat_7b_qlora_cmed2_e3 .py 

# 模型训练后得到的pth格式参数存放的位置
#export PTH=/root/cmed2/config/work_dirs/ internlm2_chat_7b_qlora_cmed2_e3 /epoch_3.pth

export PTH=/root/cmed2/config/work_dirs/internlm2_chat_7b_qlora_cmed2_e3/iter_1014.pth

# pth文件转换为Hugging Face格式后参数存放的位置
export SAVE_PATH=/root/cmed2/config/work_dirs/hf

# 执行参数转换
xtuner convert pth_to_hf $CONFIG_NAME_OR_PATH $PTH $SAVE_PATH
```



### 6.2 模型合并：将微调后的模型参数合并到基座模型中

-   用以下命令行
    ```bash
    # 建议命令不要换行，不易出错：
    xtuner convert merge /root/model/Shanghai_AI_Laboratory/ internlm2-chat-7b   /root/cmed2/config/work_dirs/hf   /root/cmed2/config/work_dirs/hf_merge  --max-shard-size 2GB
    ```

## 7. 模型试用

### 7.1 安装网页Demo所需依赖

```bash
pip install streamlit==1.24.0
```

### 7.2 下载InternLM项目代码：

```bash
# 创建code文件夹用于存放InternLM项目代码
cd /root/cmed2/code
git clone https://github.com/InternLM/InternLM.git
```

### 7.3 启动网页Demo

-   1、端口映射：保持这个powershell窗口不要关闭
    ```bash
    # 端口要改成自己的
    ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 36846

    #ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 36979
    ```
-   **2、启动网页（使用的是微调合并后的模型）：**
    -   1-1、复制chat/web\_demo.py为web\_demo\_finetune.py，并修改配置：
        ```bash
        model = (AutoModelForCausalLM.from_pretrained('/root/cmed2/config/work_dirs/hf_merge',
                                                          trust_remote_code=True).to(
                                                              torch.bfloat16).cuda())
        tokenizer = AutoTokenizer.from_pretrained('/root/cmed2/config/work_dirs/hf_merge',
                                                      trust_remote_code=True)
        ```
        
    -   **1-2、web\_demo\_finetune.py修改下面的内容（用绝对路径），否则会报错：**
        ```bash
        user_avator = '/root/cmed2/code/InternLM/assets/user.png'
        robot_avator = '/root/cmed2/code/InternLM/assets/robot.png'
        ```
        
    -   1-3、启动web_demo_finetune页面：
        ```bash
        streamlit run /root/cmed2/code/InternLM/chat/web_demo_finetune.py --server.address 127.0.0.1 --server.port 6006
        ```
        

### 7.4 访问网页Demo,效果演示

在浏览器中输入：[http://localhost:6006](http://localhost:6006 "http://localhost:6006")


### 7.5 以基座模型，启用web\_demo页面：【用于对比】

-   修改路径为下面这样（用绝对路径），否则会报错：
    ```bash
    user_avator = '/root/cmed/code/InternLM/assets/user.png'
    robot_avator = '/root/cmed/code/InternLM/assets/robot.png'
    ```

```bash
streamlit run /root/cmed2/code/InternLM/chat/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

-   效果演示
    > 在浏览器中输入：[http://localhost:6006](http://localhost:6006 "http://localhost:6006")![](image/image_zJcnleEdR-.png)



## 8. 模型发布到OpenXlab

### 8.1 模型发布：在云服务器中通过openxlab包直接上传，发布到OpenXlab

-   **1：获取openxlab账号的Access key ID 和 Secret Access Key**

-   **2：安装openxlab 库，并进行认证**
    ```bash
    python -m pip install -U openxlab

    ```
    -   **认证方法：**
        > 编写python代码**openxlab\_login.py**，并填写自己的access key 和 secret key。
        ```纯文本
        import openxlab
        openxlab.login(ak=<Access Key>, sk=<Secrete Key>)
        ```
        -   **执行openxlab_login.py:**
            ```bash
            cd /root/cmed2/code/internlm2-chat-cmed/openxlab

            python openxlab_login.py
            ```
        
-   **3：在模型存放路径，编写模型元文件**
    -   **想要上传这个模型，就需要先编写一个元文件**：

        本文有个适用于InternLM2-chat-7b结构的快速方法。因为文件夹结构和原来InternLM2-chat-7b结构一样，而模型中心里可以**直接复制一份已有的元文件：**
        -   先在模型中心获取internlm2-chat-7b 的元文件，链接为：[https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b](https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b "https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b")
        -   查看元信息：
        -   **将这个元文件的内容，复制另存为：metafile.yml**
    -   **要想基于该元文件创建并上传模型文件，还需要对每个文件补充Weights键值**
        -   **如何快速生成Weights信息呢？**

            要将获取的该模型元文件放到模型路径的同一级目录下，下面是快速生成Weights的脚本gen_weights.py，代码如下：
            ```bash
            import sys
            import ruamel.yaml

            yaml = ruamel.yaml.YAML()
            yaml.preserve_quotes = True
            yaml.default_flow_style = False
             file_path = 'metafile.yml' 
            # 读取YAML文件内容
            with open(file_path, 'r') as file:
             data = yaml.load(file)
            # 遍历模型列表
            for model in data.get('Models', []):
             # 为每个模型添加Weights键值对，确保名称被正确引用
             model['Weights'] = model['Name']

            # 将修改后的数据写回文件
            with open(file_path, 'w') as file:
             yaml.dump(data, file)

            print("Modifications saved to the file.")
            ```
        
-   **4：更新模型元文件，并上传模型到openxlab**
    -   **假设，模型文件夹是hf_merge, 元文件是metafile.yml：**
        -   **上传前，要先登录openxlab**
            ```python
            cd /root/cmed2/code/internlm2-chat-cmed/openxlab

            #python openxlab_login.py
            python login.py

            ```
        -   将相关文件复制到模型路径的同一级目录hf_merge中，**并更新元文件metafile.yml**
            
            ```bash
            # 将生成的模型的元文件metafile.yml放到模型路径的同一级目录下：
            cp /root/cmed2/code/internlm2-chat-cmed/openxlab/metafile.yml /root/cmed2/config/work_dirs/hf_merge
            # 将快速生成weights的文件放到模型路径的同一级目录下：
            cp /root/cmed2/code/internlm2-chat-cmed/openxlab/gen_weights.py /root/cmed2/config/work_dirs/hf_merge

            cd /root/cmed2/config/work_dirs/hf_merge

            # 快速生成weights的文件（即更新metafile.yml）
            python gen_weights.py

            ```
            > **gen\_weights.py**执行后，元文件metafile.yml将更新每一个文件的[文件名](https://so.csdn.net/so/search?q=文件名\&spm=1001.2101.3001.7020 "文件名")到weight 字段中。
        -   **将自个微调合并后的模型上传到openxlab**
            > 用户名就是自己的openxlab用户名，这个要去平台上查看。模型仓库名是要建立的新模型仓库名字，符合要求和现有的不重复就可以随便起一个。 &#x20;
            > 注：如果上传失败，需要在openxlab上将模型中的文件全部删除后，才能再重新上传。
            ```python
            cd /root/cmed2/config/work_dirs/hf_merge

            # 将自个微调合并后的模型上传到openxlab
            #openxlab model create --model-repo='用户名/模型仓库名'  -s  ./metafile.yml
            openxlab model create --model-repo='kofancy/internlm2-chat-cmed'  -s  /root/cmed2/code/internlm2-chat-cmed/openxlab/metafile.yml
            ```
       
    -   **上传后看到openXlab上已经我们的模型了：**

        [https://openxlab.org.cn/models/detail/kofancy/internlm2-chat-cmed](https://openxlab.org.cn/models/detail/kofancy/internlm2-chat-cmed "https://openxlab.org.cn/models/detail/kofancy/internlm2-chat-cmed")

    
        
## 9.应用部署到openxlab

### 9.1 量化

参考：[https://huggingface.co/lmdeploy/llama2-chat-7b-w4/blob/main/README\_zh-CN.md](https://huggingface.co/lmdeploy/llama2-chat-7b-w4/blob/main/README_zh-CN.md "https://huggingface.co/lmdeploy/llama2-chat-7b-w4/blob/main/README_zh-CN.md")

-   首先安装LmDeploy
    ```bash
    pip install -U lmdeploy

    ```
-   模型转换（采用离线转换的方式）：
    > 使用 TurboMind 推理模型需要先将模型转化为 TurboMind 的格式。
    > 离线转换需要在启动服务之前，将模型转为 lmdeploy TurboMind 的格式，如下所示。
    ```python
    # 转换模型（FastTransformer格式） TurboMind
    # lmdeploy convert 模型名称  模型路径
    lmdeploy convert internlm2-chat-7b ./internlm2-chat-cmed
    ```
    -   执行完成后将会在当前目录生成一个 `workspace` 的文件夹。这里面包含的就是 TurboMind 和 Triton “模型推理”需要到的文件。
        > 产生一个 `workspace` 文件夹，将其重命名。
        ```python
        mv workspace cmed_workspace
        ```
-   **然后转换模型为turbomind格式**
    -   \--dst-path: 可以指定转换后的模型存储位置。
    -   **lmdeploy convert internlm-chat-7b  要转化的模型地址 --dst-path 转换后的模型地址**
    -   LmDeploy Chat 对话
        lmdeploy chat turbomind 转换后的turbomind模型地址
    ```bash
    mkdir -p /root/cmed2/deploy && cd /root/cmed2/deploy
    # 不成功
    lmdeploy convert internlm2-chat-7b  /root/cmed2/config/work_dirs/hf_merge/ --dst-path /root/cmed2/deploy

    #lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/ --dst-path /root/cmed/deploy 
    ```
-   lmdeploy 还支持 Python 直接与 TurboMind 进行交互

-   使用lmdeploy开启服务：
    ```python
    lmdeploy serve api_server cmed_workspace --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
    ```
    ```python
    #悟空-Chat 启动
    lmdeploy serve api_server swk_workspace --server-name ${gradio_ui_ip} --server-port ${gradio_ui_port}
    ```

## 10. 模型评测

[ OpenCompass  https://opencompass.org.cn/doc](https://opencompass.org.cn/doc " OpenCompass  https://opencompass.org.cn/doc")

-   安装 OpenCompass：
    ```bash
    cd /root/cmed2/code

    git clone https://github.com/open-compass/opencompass
    cd opencompass

    pip install -e .
    ```
-   下载解压数据集：
    ```bash
    cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
    unzip OpenCompassData-core-20231110.zip

    ```
    -   法二：
        ```bash
        # Download dataset to data/ folder
        wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
        unzip OpenCompassData-core-20231110.zip
        ```
-   评测启动：
  -   样例：

      [**https://opencompass.org.cn/doc**](https://opencompass.org.cn/doc "https://opencompass.org.cn/doc")
      ```bash
      python run.py --datasets siqa_gen winograd_ppl \
      --hf-path facebook/opt-125m \  # HuggingFace 模型路径
      --tokenizer-path facebook/opt-125m \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
      --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
      --model-kwargs device_map='auto' \  # 构建模型的参数
      --max-seq-len 2048 \  # 模型可以接受的最大序列长度
      --max-out-len 100 \  # 生成的最大 token 数
      --batch-size 64  \  # 批量大小
      --num-gpus 1  # 运行模型所需的 GPU 数量
      ```
      -   确保OpenCompass正确安装并准备好数据集后，您可以使用以下命令评估LLaMA-7b模型在MMLU和C-Eval数据集上的性能：
          ```bash
          python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl
          ```
      -   通过命令行评估其他 HuggingFace 模型。以LLaMA-7b为例：
          ```bash
          python run.py --datasets ceval_ppl mmlu_ppl \
          --hf-path huggyllama/llama-7b \  # HuggingFace model path
          --model-kwargs device_map='auto' \  # Arguments for model construction
          --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # Arguments for tokenizer construction
          --max-out-len 100 \  # Maximum number of tokens generated
          --max-seq-len 2048 \  # Maximum sequence length the model can accept
          --batch-size 8 \  # Batch size
          --no-batch-padding \  # Don't enable batch padding, infer through for loop to avoid performance loss
          --num-gpus 1  # Number of minimum required GPUs
          ```
      ```bash
      python run.py --datasets ceval_ppl mmlu_ppl \
      --hf-path  kofancy/internlm2-chat-cmed  \  # HuggingFace model path
      --model-kwargs device_map='auto'  trust_remote_code=True  \  # Arguments for model construction
      --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False  trust_remote_code=True  \  # Arguments for tokenizer construction
      --max-out-len 100 \  # Maximum number of tokens generated
      --max-seq-len 2048 \  # Maximum sequence length the model can accept
      -- batch-size 2  \  # Batch size
      --no-batch-padding \  # Don't enable batch padding, infer through for loop to avoid performance loss
      --num-gpus 1  # Number of minimum required GPUs
       --debug
      ```
  