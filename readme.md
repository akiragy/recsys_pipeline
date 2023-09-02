Build Recommender System with PyTorch + Redis + Elasticsearch + Feast + Triton + Flask. Term / Vector Recall, DeepFM Ranking, Inference Engine and Web Application.
=================
**English | [中文](./readme-cn.md)**

Using the well-known MovieLens dataset as an example, we will introduce the recommender system pipeline from offline to online, with all operations can be executed on a single laptop. 
Despite the utilization of multiple components, it is important to note that everything is contained within Conda and Docker, ensuring no impact on the local environment.

- In the Conda environment, pandas and PyTorch are installed to run offline data processing and model training, simulating an industrial workflow like the HDFS storage -> Spark ETL -> GPU cluster training.
- In the Conda environment, Flask is deployed as the web server, simulating an industrial Java Spring backend.
- In the Docker environment, Redis, Elasticsearch, Feast Feature Store, and Triton Inference Server are installed, simulating RPC calls from a backend (Flask) to other components (Docker). 
Specifically, Redis serves as the data store for user terms and vectors required for recall, Elasticsearch is used to create an item term index and a vector index essential for recall, 
Feast is utilized to store user and item features required for ranking, while Triton serves as a real-time prediction engine for ranking.

The overall architecture of the recommender system is illustrated below. Now we will introduce the development and deployment processes for the recall and ranking modules across three phases: offline, offline-to-online, and online.

![img.png](pic/img_head1.png)

# 1 Offline
## 1.0 Conda Environment 
```bash
conda create -n rsppl python=3.8
conda activate rsppl
conda install --file requirements.txt --channel anaconda --channel conda-forge
```

## 1.1 Preprocessing
```bash
cd offline/preprocess/
python s1_data_split.py
python s2_term_trans.py
```
Preprocessing of labels, samples, and features:

- Labels: In the field of recommender systems, there is a prevalence of implicit feedback (click/no click) over explicit feedback (ratings). Consequently, we transformed the MovieLens dataset into an implicit feedback dataset: 
ratings greater than 3 were labeled as positive samples, while ratings less than or equal to 3 were labeled as negative samples. In total, there are 575,281 positive samples and 424,928 negative samples.
- Samples: We sorted the rating actions of each user based on timestamp. For each user, the last 10 rated items were reserved for online evaluation, totaling 60,400 samples. 
The remaining samples were split such that the first 80% of rated items for each user are used as the offline training set (totaling 754,233 samples), while the last 20% are used as the offline test set (totaling 185,576 samples).
- Features: Sparse features were encoded into integer sequences starting from 0.

It is important to highlight the concept of point-in-time joins in the s2_term_trans.py file. 
Specifically, during the generation of offline training samples (imp_term.pkl), features up to and including the time of the action closest to the present moment should be utilized.
Using features after this point would introduce feature leakage, whereas using features that are significantly distant in time from the present would result in inconsistencies between offline and online. 
In contrast, the most recent features should be deployed for online serving (user_term.pkl and item_term.pkl).

## 1.2 Term / Vector Recall
```bash
cd offline/recall/
python s1_term_recall.py
python s2_vector_recall.py
```
Term recall: This component utilizes the user's past interactions with items(' genres) within a specified time window to match user preferences with items. These terms will be loaded into Redis and Elasticsearch later.

Vector recall: In this component, FM (Factorization Machines) are employed, utilizing only the user ID and item ID as features. The resulting AUC (Area Under the Curve) = 0.8081. 
Upon completion of the training phase, user and item vectors are extracted from the model checkpoint and will be loaded into Redis and Elasticsearch for subsequent utilization.

## 1.3 Feature Engineering and DeepFM Ranking
```bash
cd offline/rank/
python s1_feature_engi.py
python s2_model_train.py
```
In total, there are 59 features including three types: one-hot features (such as userid, itemid, gender, etc.), multi-hot features (genres), and dense features (historical behavioral statistics).

The ranking model employed is DeepFM, achieving an AUC of 0.8206. While [pytorch-fm](https://github.com/rixwew/pytorch-fm) is an elegant package for FM-based algorithms, 
it has two limitations: 1. It exclusively supports sparse features and lacks support for dense features. 2. All sparse features share the same dimension, 
which violates the intuition that "ID embeddings should be high-dimensional, while side info embeddings should be low-dimensional.". 
To address these constraints, we made modifications to the source code, enabling support for both dense features and varying embedding dimensions for sparse features. 
Additionally, it was noted that the deep embedding module negatively impacted model performance, which prompted its removal from the model. 
As a result, the current model structure primarily consists of a sparse FM module integrated with a dense MLP, it is not a conventional DeepFM.

# 2 Offline2Online
## 2.0 Docker Images
To prevent any impact on the local environment, Redis, Elasticsearch, Feast and Triton are all employed within Docker containers. 
To proceed, please ensure that [Docker](https://www.docker.com/) has been installed on your laptop, and then execute the following commands to download the respective Docker images.
```bash
docker pull redis:6.0.0
docker pull elasticsearch:8.8.0
docker pull feastdev/feature-server:0.31.0
docker pull nvcr.io/nvidia/tritonserver:20.12-py3
```

## 2.1 User Terms and Vectors to Redis
[Redis](https://redis.io/) is utilized as a database for storing user information needed for recall.

Start the Redis container.
```bash
docker run --name redis -p 6379:6379 -d redis:6.0.0
```
The process involves loading the user's term, vector, and filter into Redis. The term and vector data are generated in section 1.2, while the filter pertains to items with which the user has previously interacted. 
These filtered items are excluded when generating recommendations.

Once the data is loaded, a validation step will be carried out by checking the data for a sample user. 
A successful validation will be indicated by the output displayed below.
```bash
cd offline_to_online/recall/
python s1_user_to_redis.py
```

![img_redis1.png](pic/img_redis1.png)

## 2.2 Item Terms and Vectors to Elasticsearch
[Elasticsearch](https://www.elastic.co/) is employed to create both inverted index and vector index for items. Originally designed for search applications, Elasticsearch's fundamental use case involves using words to retrieve documents. 
In the context of recommender systems, we treat an item as a document, and its terms, such as movie genres, as words. This allows us to use Elasticsearch to retrieve items based on these terms. 
This concept aligns with the notion of inverted indexes, making Elasticsearch a valuable tool for term recall in recommender systems.
For vector recall, a commonly used tool is Facebook's open-sourced faiss. However, for the sake of ease of integration, we have opted to utilize Elasticsearch's vector retrieval capabilities.
Elasticsearch has supported vector retrieval since version 7, and approximate K-nearest neighbor (ANN) retrieval since version 8.
In this project, we install a version of Elasticsearch that is 8 (or higher). This choice is made because precise KNN retrieval often cannot meet the low-latency requirements of an online system.

Start the Elasticsearch container and enter its internal terminal by executing following commands.
```bash
docker run --name es8 -p 9200:9200 -it elasticsearch:8.8.0
```
Copy the password displayed in your terminal output (as indicated below) and paste it as the value for ES_KEY in the data_exchange_center/constants.py file. 
This step is necessary because Elasticsearch has implemented password authentication requirements starting from version 8.
![img_es1.png](pic/img_es1.png)  

Once the password has been pasted, exit the internal terminal by using the Ctrl+C (or Command+C) keyboard shortcut. 
This action will also stop the container, so we need to restart the Elasticsearch container and ensure it runs in the background as part of the subsequent steps.
```bash
docker start es8
```
Load the item terms to create the term index and load the item vectors to create the vector index. In industrial settings, these two indexes are typically separated for better performance and flexibility. 
However, for simplicity in this project, we are combining them into a single index.

Following the data loading process, a validation step will be carried out by checking a sample item's term and vector. 
A successful validation will be indicated by the output displayed below.
```bash
cd offline_to_online/recall/
python s2_item_to_es.py
```

![img_es2.png](pic/img_es2.png)

## 2.3 User and Item Features to Feast
[Feast](https://feast.dev/) stands as the pioneering open-source feature store, holding historical significance in this domain. Feast includes both offline and online components. 
The offline component primarily facilitates point-in-time joins. However, since we have managed point-in-time joins in pandas ourselves, there is no necessity to utilize Feast's offline capabilities. 
Instead, we employ Feast as an online feature store. 
The reason behind not using Feast for point-in-time joins lies in the fact that Feast primarily offers feature storage capabilities and 
lacks feature engineering capabilities (although it has introduced some basic transformations in recent versions). 
Most companies prefer customized feature engines with more powerful capabilities. Therefore, there is no need to invest much effort in learning Feast's offline usage.
It is more practical to employ more general tools such as pandas or Spark for feature processing and leverage Feast solely as a component for transporting features between offline and online.

Convert the feature files from CSV to Parquet format to meet Feast's requirements.
```bash
cd offline_to_online/rank/
python s1_feature_to_feast.py
```
Open a new terminal (called terminal 2), start the Feast container with 6566 as the HTTP port and enter its internal terminal by executing following commands.
```bash
cd data_exchange_center/online/feast
docker run --rm --name feast-server --entrypoint "bash" -v $(pwd):/home/hs -p 6566:6566 -it feastdev/feature-server:0.31.0
```
Execute following commands in the Feast container's internal terminal.
```bash
# Enter the config directory
cd /home/hs/ml/feature_repo
# Initialize the feature store from config files (read parquet to build database)
feast apply
# Load features from offline to online (defaults to sqlite, with options like Redis available)
feast materialize-incremental "$(date +'%Y-%m-%d %H:%M:%S')"
# Start the feature server
feast serve -h 0.0.0.0 -p 6566
```
After completing all the steps, the following output will be displayed.

![img.png](pic/img_feast.png)  

Back to terminal 1, execute the following command to test if Feast is serving properly. A response json string will be printed if successfully.
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

## 2.4 PyTorch to ONNX to Triton
[Triton(Triton Inference Server)](https://developer.nvidia.com/triton-inference-server) is an open-source inference serving engine developed by Nvidia.
Triton Inference Server offers support for a diverse range of frameworks such as TensorFlow, PyTorch, ONNX, and additional options, making it an excellent choice for model serving.
Despite it's developed by Nvidia, Triton is versatile enough to serve with CPUs, offering flexibility in its usage.
While a more prevalent industry solution involves TensorFlow -> SavedModel -> TensorFlow Serving, Triton is gaining popularity due to its adaptability in switching between different frameworks. 
Hence, in this project, we adopt a pipeline that utilizes PyTorch -> ONNX -> Triton Server.

Convert the PyTorch model to ONNX format.
```bash
cd offline_to_online/rank/
python s2_model_to_triton.py
```
Open a new terminal (called terminal 3), start the Triton container with 8000 as the HTTP port and 8001 as the gRPC port by executing following commands.
```bash
cd data_exchange_center/online/triton
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/:/models/ nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models/
```

![img_triton2.png](pic/img_triton2.png)
  
Back to terminal 1, run the script to test the consistency between offline and online prediction scores. This step will help ensure the reliability of the recommender system.
```bash
cd offline_to_online/rank/
python s3_check_offline_and_online.py
```    
As shown below, the offline and online scores are identical, indicating the consistency between offline and online.

![img_triton1.png](pic/img_triton1.png)


# 3 Online
## 3.1 Server
In industrial settings, engineers commonly opt for Java + SpringBoot or Go + Gin as the backend for recommender systems.
However, in this project for the sake of integration ease, Python + Flask is utilized.
It's worth noting that there are several web frameworks for Python, including Django, Flask, FastAPI, and Tornado, 
all of which are capable of routing RestAPI requests to functions for processing.
Any of these frameworks could meet our requirements, Flask was selected randomly for this project.

Open a new terminal (called terminal 4), start the Flask web server with 5000 as the HTTP port by executing following commands.
```bash
conda activate rsppl
cd online/main
flask --app s1_server.py run --host=0.0.0.0
```

![img_flask1.png](pic/img_flask1.png) 

## 3.2 Client
Back to terminal 1 and conduct a test by calling the recommendation service from a client (in this context, the client refers to the upstream service that calls the recommendation service, not user devices). 
The results will be returned in JSON format, with the top 50 recommended item ids displayed.
Subsequently, the downstream service can use these item ids to retrieve their corresponding attributes and then provide them to the client (in this context, user device). 
This completes a full recommendation flow.

```bash
cd online/main
python s2_client.py
```

![img_flask2.png](pic/img_flask2.png)