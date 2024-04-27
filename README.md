# Predicting pKi Using Transformer and CNN

**Primary location of analysis and results in `challenge.ipynb`**

As my submission for this challenge, I have compared two approaches head to head for predicting pKi in 4 different kinases using SMILES input strings. As a simple model and baseline, I created a CNN that convolves over a one-hot-encoded version of the SMILES data. I trained one model for each kinase, and then, a unified model for all kinases, which involved adding kinase identity to the one-hot input encoding.     

Next, I built an encoder-only transformer with a dense output layer. This model proved to be the best at predicting pKi for all kinases (shown below). The notebook `challenge.ipynb` contains more in-depth explanation of my thought process, and code for plotting results. Models are implemented in `models.py`, while `data_utils.py` & `notebook_utils.py` are mostly boring stuff that I moved out to avoid cluttering the notebook.

![](./plot.png?raw=true)
