# BERTSuicdeClassify

I implemented BERT model to classify suicide text from Reddit platform. Data can be found at [link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).

## I. Set up environment

Git clone this repo:
```
git clone https://github.com/hoangcaobao/BERTSuicdeClassify
```
Change directory to repo:
```
cd BERTSuicdeClassify
```
Install necessary libraries:
```
pip3 install -r requirements.txt
```
## II. Set up data
Go to Kaggle [link](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) to download csv file
Move the csv file to folder

## III. Train model
Run train.py file to train model:
```
python3 train.py --epochs ${training epochs}
```
For example:
```
python3 train.py --epochs 5
```

## IV. Result
After finish training, you can find history of training in both csv and png file. You also have BERT weights to apply to other uses.

## HOANG CAO BAO
