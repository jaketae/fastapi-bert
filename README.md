# CoLA BERT FastAPI

This repository demonstrates a minimal example of how to deploy a fine-tuned BERT model using FastAPI. 

## Setup

Clone the repository via

```
git clone
```

Using [conda](https://docs.conda.io/en/latest/), create a new virtual environment and install dependencies specified in `spec-file.txt` via

```
conda create --name myenv --file spec-file.txt
```

You can activate the environment any time in the terminal via

```
conda activate myenv
```

## Experiment

A total of 3 BERT and BERT-variant models were fine-tuned on the GLUE [CoLA](https://nyu-mll.github.io/CoLA/) dataset. The dataset contains sentences, each with labels indicating whether they are grammaticality acceptable or not. Hence, its formulation is a classic binary classification problem. Below is an example entry from the dataset, loaded through HuggingFace [Datasets](https://huggingface.co/docs/datasets/).

```python
{
    'idx': 0,
    'label': 1,
    'sentence': "Our friends won't buy this analysis, let alone the next one we propose."
}
```

For training and validation, we use the HuggingFace transformers [Trainer API](https://huggingface.co/transformers/main_classes/trainer.html) to expedite prototyping. Since one of the tertiary goals of this experiment is to determine how different models compare to each other in minimally fine-tuned conditions, we do not delve into hyperparameter search.

The experiment can be run by simply running all the cells of the Jupyter notebook. Note that without running the experiment, it will not be possible to spin up the FastAPI web application since it won't have any reference point to load model weights from.

## Result

Below is a summary of the results of the experiment, seeded at 42. The specific training arguments can be accessed in `experiment.ipynb`, where Colab notebook in which the experiment was conducted. [Matthew's correlation](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient) was used as a performance metric.


|         | bert-base-uncased | distilbert-base-uncased | distilroberta-base |
|:-------:|:-----------------:|:-----------------------:|:------------------:|
| Epoch 1 |       0.521       |          0.449          |        0.361       |
| Epoch 2 |       0.535       |          0.453          |        0.466       |
| Epoch 3 |       0.572       |          0.497          |        0.527       |
| Epoch 4 |       0.555       |          0.510          |        0.551       |
| Epoch 5 |       0.557       |          0.483          |        0.536       |

The best model, BERT-base-uncased at the third epoch, was saved to be loaded in the FastAPI app. 

`main.py` contains code relevant to spinning up the main FastAPI process; `backend.py` demonstrates how a trained model can be loaded into memory and run for inference. To optimize serving, we perform basic quantizing by using  `torch.qint8` for `nn.Linear` layers in the BERT model.

## Demo

Activate the appropriate conda environment, `cd` into the repository directory, then run the FastAPI app by typing

```
uvicorn main:app --reload
```

As the purpose of this project was to demonstrate a minimally functional serving, the app does not have a user-facing frontend. Instead, to run model inference, one can access the app via the `curl` command, as follows:

```
curl -H "Content-type: application/json" -X POST -d '{"passage":"I are a boy"}' http://localhost:8000/generate
```

The following is a sample response.

```
{"prob": 0.004417025949805975}
```

