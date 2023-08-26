# IDP-LM: prediction of protein intrinsic disorder and disorder functions based on language models

This repository contains the source code used in our paper titled [Yihe Pang, Bin Liu. IDP-LM: prediction of protein intrinsic disorder and disorder functions based on language models]. The code is implemented to realize the predictors proposed in the paper and includes examples and tutorials to assist users in utilizing the code.

## Citation
Upon the usage the users are requested to use the following citation:
[Yihe Pang, Bin Liu. IDP-LM: prediction of protein intrinsic disorder and disorder functions based on language models. (Submitted)]

## Introduction
We proposed a disorder specific protein language model, IDP-BERT. The IDP-BERT was trained as a Restrictive Masked Language Model (ReMLM) to focus on the disordered regions mainly located in the N’ and C’ terminals of sequences. Furthermore, we proposed a computational predictor called IDP-LM for predicting intrinsic disorder and disorder functions by leveraging the pre-trained protein language models. IDP-LM takes the embeddings extracted from three pre-trained protein language models as the exclusive inputs, including ProtBERT, ProtT5, and IDP-BERT. The evaluation results on independent test datasets demonstrated that the IDP-LM provided high-quality prediction results for intrinsic disorder and four common disordered functions including disorder protein binding, DNA binding, RNA binding, and disorder flexible linkers.


## Acknowledgments
We acknowledge with thanks the following databases and softwares used in this server:<br> 
DisProt(https://www.disprot.org/): database of intrinsically disordered proteins.<br> 
MobiDB (https://mobidb.bio.unipd.it/): database of protein disorder and mobility annotations.<br> 
PDB (https://www.rcsb.org/): RCSB Protein Data Bank.<br> 
ProtTrans (https://github.com/agemagician/ProtTrans): Protein pre-trained language models.<br> 
