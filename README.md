## Latent Dirichlet Allocation (Python)
To understand what is Latent Dirichlet Allocation: [read here](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

Scripts created for **supervised learning**
[What is supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)

## Library Required
* numpy
* nltk

## Data Cleaning
* Lemmatization from (NLTK)
* Stop words
---
## Create data set (Mac - Terminal)
1. Create "build" directory in the same folder
```
    mkdir build
```
2. Create text file inside "build" directory (Eg. food.txt)
3. Content of food text file should be as follow
Format of content
```
"<KEYWORDS"|"<CLASS TYPE>"
```
Example
```
"Cabbage Celery Chicory Corn"|"Vegetable"
"Beef Chicken Fish"|"Meat"
```
4. Type the following command to train the model
```
python classifier.py -t -f food.txt -m food
```
5. Two file will be created in build folder:
    * stopwords.p
    * food_trained.p
6. Result of perplexity and keyword weightage will be display in terminal

---

## How to classify after train (Mac - Terminal)
1. In terminal type
```
python classifier.py -c -m food -l "beef"
```
2. Result will be shown in terminal

---

## Check commands (Mac - Terminal)
1. In terminal type
```
python classifier.py --sos
```

## Commands
| Command | Type | Default Value | function | Example |
|--- | --- | --- | --- | --- |
|--alpha | float | 0.005 | Alpha value |--alpha 0.001 |
|--beta | float | 0.005 | Beta value |--beta 0.001 |
|-t | boolean | false | Trigger train model function | -t |
|-c | boolean | false | Trigger classification function | -c |
|-m | string | no default value | Model name | -m food |
|-l | string | no default value | String to pass in for classification| -c "Beef Chicken" |
|-k | integer | 10 | Number of topics | -k 50 |
|-i | integer | 100 | Number of iteration | -i 200 |
|-f | string | no default value| Filename | -f food.txt |
|--sos | boolean | false | Display example and command list| --sos |

---

## Reference
* LDA SOURCE: https://github.com/shuyo/iir/blob/master/lda/llda.py  
