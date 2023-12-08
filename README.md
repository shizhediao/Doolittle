# Doolittle: Benchmarks and Corpora for Academic Writing Formalization
Source code for the EMNLP 2023 paper entitled "Doolittle: Benchmarks and Corpora for Academic Writing Formalization" by Shizhe Diao et al.

## *Doolittle* Dataset for Academic Writing Formalization (AWF) Task
Improving the quality of academic writing is a meaningful but challenging task. 
Conventional methods of language refinement focus on narrow, specific linguistic features within isolated sentences, such as grammatical errors and improper word use. 
We propose a more general task, `Academic Writing Formalization (AWF)`, to improve the overall quality of formal academic writing at the paragraph level. 
We formulate this language refinement task as a formal text style transfer task which transfers informal-academic text to formal-academic and contribute a large-scale non-parallel dataset, *`Doolittle`*, for this purpose.

### Dataset Description
*`Doolittle`* is a large-scale non-parallel dataset for AWF task. 
It contains 13,000 training samples and 465 dev samples for each of the two domains, informal-academic and formal-academic.
Please request access to Doolittle dataset by filling in this [form](x) and we will send you the download link via email.

Then please put the full dataset under [`AWF-dataset/`](.AWF-dataset/) folder. 
The detailed information is:
|Description|File Name|#Paragraphs |Parallel|
|:--------:| :---------:|:--------:|:--------:|
| Informal-academic train set | paragraph_native_train.0 |13.0K|No|
| Formal-academic train set | paragraph_native_train.1 |55.6K|No|
| Informal-academic dev set | paragraph_native_dev.0 |465|Yes|
| Formal-academic dev set | paragraph_native_dev.1 |465|Yes|
| Informal-academic test set | paragraph_native_test.0 |415|Yes|
| Formal-academic test set | paragraph_native_test.1 |415|Yes|
| Informal-academic dev set for MORL Training | dev.0.csv |465|No|

## Metric-Oriented Reinforcement Learning (MORL)

To address our task with reduced cost and better performance, we propose a method called `Metric-Oriented Reinforcement Learning (MORL)`. 
This methodology, inspired by Reinforcement Learning with Human Feedback (RLHF), follows a three-step training process:

**Step 1:** Train a policy model (usually a PLM) that can meet the requirements of a task. 

**Step 2:** Select some metrics that can accurately evaluate the quality of how the task has been performed. Build a reward model that can score a given policy model’s output with a scalar. 

**Step 3:** Optimize the policy against the reward model using reinforcement learning with the proximal policy optimization (PPO) algorithm.

In our work, we chose `Galactica-1.3B` and `BART-Large` as two backbone policy models for their inherent capability in solving academic-related grammatical error correction (GEC) task. 
And used MORL to tune these two models against 4 automatic metrics which are `Transfer Accuracy (ACC)`, `Perplexity (PPL)`, `Semantic Similarity (SIM)`, `BART Score (BARTS)`. 
The code is implemented with reference to many other repositories, which are listed below:

|Code Implementation|Link to Reference Repository|
|:--------:| :---------:|
| Automatic Metrics: PPL | https://github.com/huggingface/evaluate |
| Automatic Metrics: SIM | https://github.com/martiansideofthemoon/style-transfer-paraphrase.git |
| Automatic Metrics: BARTScore | https://github.com/neulab/BARTScore |
| Reinforment Learning Algorithm: PPO | https://github.com/huggingface/trl |

We provided two notebooks as MORL tuning examples, where [MORL-BARTLarge.ipynb](MORL-BARTLarge.ipynb) tuned a `BART-Large` model while [MORL-Galactica.ipynb](MORL-Galactica.ipynb) tuned a `Galactica-1.3B` model.

## Setup

### Required Environment

```
Python = 3.10 
CUDA = 11.7
Ubuntu = 20.04
```

### Install Needed Packages
```shell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117 
cd trl
pip install -e .
```
### Tune Policy Model:
`Galactica-1.3B` and `BART-Large` are chosen as two policy models in our work. In this repository, we did not provide either the tuned policy models or the scripts to fine-tune them, please refer to our paper and [Huggingface Transformers tutorial](https://huggingface.co/docs/transformers/index) to train your own policy models on our *`Doolittle`* dataset.

### Then you can follow instructions in notebooks to start your MORL tuning :)

## Contact information

For help or issues using MORL, please submit a GitHub issue.

For personal communication related to *Doolittle* dataset and MORL, please contact Shizhe Diao (`sdiaoaa@connect.ust.hk`) or Yongyu Lei (`yleiah@connect.ust.hk`).

## Citation

If you use or extend our work, please cite the following [paper]():
```
@article{diaodoolittle,
  title={Doolittle: Benchmarks and Corpora for Academic Writing Formalization},
  author={Diao, Shizhe and Lei, Yongyu and Pan, Liangming and Fang, Tianqing and Zhou, Wangchunshu and Keh, Sedrick Scott and Kan, Min-Yen and Zhang, Tong},
  booktitle = "The 2023 Conference on Empirical Methods in Natural Language Processing",
  year={2023}
}
```