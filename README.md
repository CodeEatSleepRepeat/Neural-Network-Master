# Neural-Network-Master
Project for a master's course "Neural Networks" (MSc of Electrical Engineering and Computer Science, University of Novi Sad).

The aim of the project was to create a neural network that would be able to transform the legal questions posed in natural language to SPARQL query.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
Machine that was used in the creation of the project was equipped with Intel i7-9750H, NVIDIA GeForce RTX 2060, 16GB RAM and Windows 10 Pro.  

### Prerequisites

What things you need to install the software

```
python >= 3.6 (project was tested on Python 3.7)
pip >= 18.1 (you can probably use a different version)
```

### Installing

A step by step guide to get a development env running

1. Download the Repository from GitHub
2. Open terminal in your Linux/Windows
3. pip install all from requirements.txt like this
```
pip install -r /path/to/requirements.txt
```

## Run a project

Navigate to Repository file you just downloaded using terminal and run 'run_model.py' or other Python scripts from terminal like this:
```
python run_model.py chmod +x
```
or open your prefered code editing software that can run python code, import downloaded project into it and run it.

Repository consits of:
1. 'data' folder - contains datasets (legal questions in NLP and SPARQL format) used to train the models
2. 'model' folder - contains encoder-decoder network for running the model without training
3. 'training_images' folder - contains images that show training progress and results of model in 'model' folder
4. 'create_unique_dataset.py' - receives a dataset containing duplicate questions, and returns all questions without duplicates
5. 'fine_tuning.py' - uses the model from the 'model' folder created by 'transformer_model.py' to fit it with the desired dataset (in this case 'data/unique.legal' datasets)
6. 'run_model.py' - run the model in 'real world' use case
7. 'transformer_model.py' - trains the model using a dataset containing 894.499 instances of NLP-SPARQL questions retrieved from [here](https://figshare.com/articles/Question-NSpM_SPARQL_dataset_EN_/6118505)
8. 'transformer_sparql.py' - trains the model directly over a datasets of legal questions ('data/unique-legal' datasets)
9. '.0' - file used by nltk library

## Built With

* [Visual Studio Code](https://code.visualstudio.com/) - Visual Studio Code is a free source-code editor made by Microsoft for Windows.

## Project results

Results are published in PDF folder. For results of training look in 'training_images'. Inside is a set of 19 images with abbreviations in the name that represent:
* finetuning - these are the results of running 'fine_tuning.py' (after running 'transformer_model.py')
* legal - these are the results of running just 'transformer_sparql.py'
* sparql - these are the results of running 'transformer_model.py' (without any fine tuning for legal dataset)

* 1B - batch size is 1
* 64B - batch size is 64

* lrn_rate - represents learning rate curve
* top1_acc - represents F1 (top 1) accuracy
* top5_acc - represents F5 (top 5) accuracy
* train_loss - represents training loss
* val_acc - represents validation accuracy

## Authors

* **Boris Bibić** - [CodeEatSleepRepeat](https://github.com/CodeEatSleepRepeat)

## License

This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE) file for details.
The Project may be used for academic research and learning purposes without the need to obtain permission from the author.
If project or its parts are used for commercial project or any other money making project contact Author for permission.
