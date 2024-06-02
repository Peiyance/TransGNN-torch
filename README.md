# TransGNN: Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems

![python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![cuda 12.1](https://img.shields.io/badge/cuda-12.1-green.svg)


## About
- This is the code for SIGIR 2024 paper [TransGNN: Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems](https://arxiv.org/pdf/2308.14355).


## Abstract
Graph Neural Networks (GNNs) have emerged as promising solutions for collaborative filtering (CF) through the modeling of user-item interaction graphs. The nucleus of existing GNN-based recommender systems involves recursive message passing along user-item interaction edges to refine encoded embeddings. Despite their demonstrated effectiveness, current GNN-based methods encounter challenges of limited receptive fields and the presence of noisy "interest-irrelevant" connections. In contrast, Transformerbased methods excel in aggregating information adaptively and globally. Nevertheless, their application to large-scale interaction graphs is hindered by inherent complexities and challenges in capturing intricate, entangled structural information. In this paper, we propose TransGNN, a novel model that integrates Transformer and GNN layers in an alternating fashion to mutually enhance their capabilities. Specifically, TransGNN leverages Transformer layers to broaden the receptive field and disentangle information aggregation from edges, which aggregates information from more relevant nodes, thereby enhancing the message passing of GNNs. Additionally, to capture graph structure information effectively, positional encoding is meticulously designed and integrated into GNN layers to encode such structural knowledge into node attributes, thus enhancing the Transformer’s performance on graphs. Efficiency considerations are also alleviated by proposing the sampling of the most relevant nodes for the Transformer, along with two efficient sample update strategies to reduce complexity. Furthermore, theoretical analysis demonstrates that TransGNN offers increased expressiveness compared to GNNs, with only a marginal increase in linear complexity. Extensive experiments on five public datasets validate the effectiveness and efficiency of TransGNN.

<img src='framework_v4.pdf' width=100% />

## Citation
Please kindly use the following BibTex if you cite our paper:
```
@article{zhang2023can,
  title={Can Transformer and GNN Help Each Other?},
  author={Zhang, Peiyan and Yan, Yuchen and Li, Chaozhuo and Wang, Senzhang and Xie, Xing and Kim, Sunghun},
  journal={arXiv preprint arXiv:2308.14355},
  year={2023}
}
```

## Table of Contents  
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#Usage)


## Requirements
The code of TransGNN are implemented and tested under the following development environment:

PyTorch:
* python=3.9.12
* torch=2.2.2
* numpy=1.22.3
* scipy=1.7.3

## Dataset
Five widely used dataset are adopted:  

- [Yelp]: This dataset contains user ratings on business venues collected from Yelp.
- [Gowalla]: It is a dataset that contains users’ check-in information for point-of-interest recommendation..
- [Tmall]: This dataset contains users’ behaviors for online shopping on Tmall.
- [Amazon-Book]: Amazon-review is a widely used dataset for product recommendation. We select Amazon-book from the collection.
- [MovieLens]: We adopt the well-established version, MovieLens 10m, which contains about 10 million ratings of 10681 movies by 71567 users.


## Usage
Please unzip the datasets first. Also you need to create the `Models/` directory. The following command lines start training and testing on the three datasets, respectively. Training and testing logs for trained models are contained in the `History/` directory. We find that TransGNN demonstrates robust performance across a variety of settings and is notably resilient to variations in hyperparameter configurations. Users can achieve strong results by employing the default settings provided.

* Yelp
```
python Main.py --data yelp 
```
* Gowalla
```
python Main.py --data gowalla 
```
* Tmall
```
python Main.py --data tmall 
```
* Amazon-Book
```
python Main.py --data amazon-book 
```
* MovieLens
```
python Main.py --data ml10m 
```
