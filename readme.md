
以推荐系统中最经典的MovieLens数据集为例，介绍推荐系统从离线到在线的全流程，所有操作都可以在一台笔记本上完成，虽然缝合的东西多，但所有东西都被封在了Conda和Docker里，不会对本地环境有任何伤害。

Conda环境安装pandas和PyTorch模拟工业界的HDFS -> Spark -> GPU集群的离线模型训练。
Conda环境安装Flask模拟工业界的Spring推荐后端。
Docker环境安装Redis + Elasticsearch + Feast Feature Store + Triton Inference Server四大件，用本机localhost调用Docker来模拟工业界的推荐后端RPC调用各个组件。其中，Redis用于存储召回所需的user标签和向量，Elasticsearch用于构建召回所需的item标签和向量索引，Feast用于存储排序所需的user和item特征，Triton用作排序所需的实时打分引擎。
整个推荐系统的架构图如下，下面将分离线、离线到在线、在线三个阶段来介绍召回和排序模块在工业级推荐系统的开发流程。
![img.png](pic/img_head1.png)

# 1 离线
## 1.0 创建conda环境并安装依赖
```bash
conda create -n rsppl python=3.8
conda activate rsppl
conda install --file requirements.txt --channel anaconda --channel conda-forge
```

## 1.1 预处理
```bash
cd offline/preprocess/
python s1_data_split.py
python s2_term_trans.py
```
在标记、样本、特征方面做了一些预处理：
- 标记：搜广推中隐式反馈推荐系统（点击/未点击）远远多于显式反馈推荐系统（评分），所以我们将ml数据集转化为隐式反馈：将评分大于3的视为正样本，评分小于等于3的视为负样本，共有575281正样本，424928负样本。

- 样本：我们将每个user的打分行为按时间戳排序，每个user最后10个评分的item用作online评估（60400个），对于剩余的样本，每个user的前80%评分的item用作offline训练集（754233个），后20%评分的item用作offline测试集（185576个）。

- 特征：工业界一般会对深度学习框架进行改造，加入hashtable来存储sparse特征的embedding，这样就不需要预先定义embedding table的shape，但这部分比较超纲了，这里还是使用pytorch原生提供的固定shape的embedding，所以要对sparse特征进行label encoder，将其转换为从0开始的自然数序列。

此外，值得注意的是s2_term_trans.py文件中的point-in-time joins，即在生产离线训练样本（imp_term.pkl）时，要使用在行为发生**之时**或**之前且离现在最近的**的特征，若使用**之后**的会出现特征泄露，**之前但离现在较远**的会产生线上线下不一致。而在特征上线时，要使用最新的特征（user_term.pkl和item_term.pkl）。

## 1.2 term和vector召回
```bash
cd offline/recall/
python s1_term_recall.py
python s2_vector_recall.py
```
term召回使用user过去一段时间窗口内行为过的item所对应的genre_list，即使用user偏好和item的genre匹配，待后续导入Redis和ES。

vector召回使用FM，仅使用userid和itemid两个特征，AUC为0.8081。训练完成后解析出user和item的embedding，待后续导入Redis和ES。

## 1.3 特征工程与DeepFM排序
```bash
cd offline/rank/
python s1_feature_engi.py
python s2_model_train.py
```
共有59维特征，分为one-hot（userid、itemid、gender等）、multi-hot（genres）、dense（历史行为统计）三类。

模型使用DeepFM，AUC为0.8206。[pytorch-fm](https://github.com/rixwew/pytorch-fm) 是一个极简的FM系列算法包，但有两个缺点：1.只支持sparse，不支持dense特征；2.所有sparse特征只能设为同样的维度，违反“id embedding应该维度高一点，side info embedding应该维度低一点”这一直觉。因此，没有直接安装使用pytorch-fm，而是基于源码进行了一些修改，使得既支持dense特征，又支持id和side info特征具有不同的embedding维度。此外，观察到deep embedding部分会使得模型效果下降，因此删掉了这一部分，所以模型结构本质是sparse FM + dense MLP，不算严格的DeepFM。

# 2 离线同步至在线
## 2.0 docker镜像下载
为了不对本地环境产生影响，Redis、Elasticsearch、Feast、Triton都是以docker的形式使用，这一步要先下载镜像（需要确保本地装了[docker](https://www.docker.com/) ）。
```bash
docker pull redis:6.0.0
docker pull elasticsearch:8.8.0
docker pull feastdev/feature-server:0.31.0
docker pull nvcr.io/nvidia/tritonserver:20.12-py3
```

## 2.1 user标签和向量进Redis
大名鼎鼎的[Redis](https://redis.io/) 就不用介绍了，我们使用它来存储召回所需的user信息。

启动Redis容器，-d参数保持后台执行。
```bash
docker run --name redis -p 6379:6379 -d redis:6.0.0
```
将user的term、vector、filter导入Redis，其中term和vector产出自1.2部分，filter是user历史行为过的item，再推荐时要过滤掉这些。

代码在导入数据后还会以某个用户为例进行数据校验，如果有下图输出，代表校验成功。
```bash
cd offline_to_online/recall/
python s1_user_to_redis.py
```
![img_redis1.png](pic/img_redis1.png)

## 2.2 item标签和向量进Elasticsearch
[Elasticsearch](https://www.elastic.co/) 的大名也是无人不知，我们使用它来为item构建倒排索引和向量索引。它最初被用于搜索领域，最原始的用法是用word去检索doc，如果我们将一个item视为一篇doc，它的标签（如电影类别）视为word，就可以借助ES来根据标签检索item，这就是倒排索引的概念，因此Elasticsearch也常被用于推荐系统的term召回模块。对于向量召回，经典工具是facebook开源的faiss，但是为了方便整合，我们在这里使用Elasticsearch提供的向量检索功能，Elasticsearch自版本7开始支持向量检索，版本8开始支持近似KNN检索算法，这里我们安装的是8及以后的版本，因为精确KNN检索的性能几乎不可能满足线上使用。

启动ES容器，-it参数进入内部终端。
```bash
docker run --name es8 -p 9200:9200 -it elasticsearch:8.8.0
```
复制下图位置的密码，粘贴为data_exchange_center/constants.py文件中ES_KEY的值，因为ES从版本8开始要强制进行密码验证。 
![img_es1.png](pic/img_es1.png)
粘贴完成后，ctrl+C（或command+C）退出内部终端，此时容器也会停止运行，所以我们需要重新启动ES容器并保持后台执行。  
```bash
docker start es8
```
将item的term导入构建标签索引，vector导入构建向量索引，工业界为了性能和灵活性会将两个索引拆分，但这里简单起见我们将索引合二为一。

代码在导入数据后还会以某个term和vector为例进行数据校验，如果有下图输出，代表校验成功。
```bash
cd offline_to_online/recall/
python s2_item_to_es.py
```
![img_es2.png](pic/img_es2.png)

## 2.3 user和item特征进Feast
[Feast](https://feast.dev/) 是第一个开源的特征仓库，有历史里程碑意义，它有离线和在线两部分功能，离线部分主要提供的功能就是point-in-time joins，因为我们在pandas中自己处理了pit，所以没必要再使用Feast的离线功能，只使用它的线上特征服务即可。关于为什么不使用Feast进行pit，是因为feast本身只有特征存储的能力，不具备特征工程的能力（最新版本支持一部分简单变换），各大公司还是喜欢自研功能更强大的特征引擎，所以不需要花太多时间去研究它的离线用法，还是用更通用的pandas或者spark处理完特征，只把它当作一个离在线特征同步器来使用吧。

将特征文件由csv转为parquet格式以满足Feast的要求。
```bash
cd offline_to_online/rank/
python s1_feature_to_feast.py
```
新启动一个终端（称为终端2），先切换到预先建好的仓库目录下，再启动docker，-it进入容器内部终端，-v将当前目录的特征配置和数据同步至docker内。
```bash
cd data_exchange_center/online/feast
docker run --rm --name feast-server --entrypoint "bash" -v $(pwd):/home/hs -p 6566:6566 -it feastdev/feature-server:0.31.0
```
下面的命令在docker容器内部终端执行。
```bash
# 进入同步目录
cd /home/hs/ml/feature_repo
# 根据配置文件初始化特征仓库（读取parquet建立db）
feast apply
# 特征由离线同步至在线（默认sqlite，可以切换Redis等）
feast materialize-incremental "$(date +'%Y-%m-%d %H:%M:%S')"
# 启动特征服务
feast serve -h 0.0.0.0 -p 6566
```
全部执行完成后，会有下图输出。  
![img.png](pic/img_feast.png)  
回到终端1输入以下命令，测试Feast是否已正常服务，如果成功会返回一串json。
```bash
curl -X POST \
  "http://localhost:6566/get-online-features" \
  -d '{
    "feature_service": "ml_item",
    "entities": {
      "itemid": [1,3,5]
    }
  }'
```

## 2.4 PyTorch转ONNX进Triton
[Triton](https://developer.nvidia.com/triton-inference-server) 全称是Triton Inference Server，是Nvidia开源的全模型serving引擎，支持TensorFlow、PyTorch、ONNX和其他各种模型，虽然是N家的产品，但是也可以使用cpu进行serving，所以请放心使用。业界更通用的方案是TensorFlow -> SavedModel -> TF Serving，但Triton因为不会被绑定在一家平台上，个人非常看好它的前景，所以这里使用的是PyTorch -> ONNX -> Triton Server的方案。

将pytorch模型导出为onnx格式。
```bash
cd offline_to_online/rank/
python s2_model_to_triton.py
```
新启动一个终端（称为终端3），先切换到预先建好的仓库目录下，再启动docker，-v将当前目录的模型同步至docker内，会有下图输出，8000是http接口，8001是grpc接口。
```bash
cd data_exchange_center/online/triton
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/:/models/ nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models/
```
![img_triton2.png](pic/img_triton2.png)
回到终端1运行以下命令，测试模型线下与线上打分是否一致。  
```bash
cd offline_to_online/rank/
python s3_check_offline_and_online.py
```
如图，线下与线下打分完全一致，测试通过。    
![img_triton1.png](pic/img_triton1.png)


# 3 在线
## 3.1 server启动
Python常用的web框架有Django, Flask, FastAPI, Tornado等，都可以实现REST API请求 -> url路由到某个函数 -> 处理的逻辑，各有优缺点，随机选取了[Flask](https://flask.palletsprojects.com/en/2.3.x/) 。这里使用Python作为后端仅仅是因为环境安装方便，工业界的推荐系统一般会使用Java + SpringBoot或者Go + Gin作为web后端。

新启动一个终端（称为终端4），输入以下命令启动flask web server，会有下图输出，服务运行在5000端口。server被调用时会输出一些日志，后续可以观察。
```bash
conda activate rsppl
cd online/main
flask --app s1_server.py run --host=0.0.0.0
```
![img_flask1.png](pic/img_flask1.png)  
## 3.2 client调用
回到终端1，测试客户端（这里的客户端不是指用户终端设备，而是指调用推荐服务的上游服务）调用，获取json格式的推荐结果，下图展示了top50的推荐列表，下游服务取这些item对应的字段并返回给客户端（用户终端），一个完整的推荐请求就结束了。
```bash
cd online/main
python s2_client.py
```
![img_flask2.png](pic/img_flask2.png)