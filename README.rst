==============
Graph Matching
==============

--------------
Models Used
--------------
1. SimGNN:
   - Encoder: 
     - Inputs: Initial one-hot encoded node embedding matrix $U \in R^{NXD}$
     - Outputs: Aggregated node Embedding Matrix $U \in R^{NXD}$
     - Uses: Neighbour Aggregation with Conv Nets (SAGE, GCN, GAT)
   - Attention Mechanism:
     - Inputs: Node Embedding Matrix $U \in R^{NXD}$
     - Outputs: Attention Weighted Graph Embedding Vector $h \in R^{D}$
     - Uses: Non linear weighted transform ($\tanh$) for context, sigmoid layers for att. weights, $\sum$ aggregate for h
   - Graph Interaction Extraction:
     - Inputs: Graph Embedding Vectors $h_{q}, h_{c} \in R^{D}$
     - Outputs: Interaction Score Vector $g \in R^{K}$, K being the depth of the NTN
     - Uses: Neural Tensor Network
   - Score Predictor:
     - Inputs: Graph Similarity Score Vector $g \in R^{K}$
     - Outputs: Graph Similarity Score s
     - Uses: Fully Connected Network

2. GMN Embed:
   - Encoder:
     - Inputs: 
       1. Initial Node Representation Matrix $U \in R^{NXD}$
       2. Initial Edge Representation Matrix $X \in R^{NXN}$
     - Outputs: Encoded Node and Edge Embedding Vectors $H^{0} \in R^{NXD}$ and $E \in R^{NXN}$
     - Uses: Multi Layer Perceptron Networks
   - Propagation:
     - Inputs: Encoded embeddings $H^{0} \in R^{NXD}$ and $E \in R^{NXN}$