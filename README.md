# NCSR Demokritos submission to PAN2016 Author Profiling Task.
Check out also our last year's submission for [PAN15](https://github.com/iit-Demokritos/pangram)

## Installation:

### Dataset:
In order to run the examples you will need to download the corpus for the author profiling task
from the PAN website:

http://pan.webis.de/clef16/pan16-web/author-profiling.html

### Requirements:

Install the requirements 

pip install -r requirements.txt


### Module:

You can also install the module if you would like to check it out from ipython.
git clone this project
cd projectfolder
pip install --user .


Package consists of a python module and scripts for:
- crossvalidating
- training
- testing

models on the PAN 2016 dataset format.

## Example usage:
```python
python cross.py -i path/to/training/dataset/pan16/english/ -n 4  
```
This will train a model on the English dataset for both the age and gender task and perform a 4-fold cross-validation on the same dataset. It will also print results.

```python
python train.py -i path/to/training/dataset/pan16/english/ -o ./models 
```
This will train a model on the English dataset and save the binary model in the folder provided by the -o flag argument.
```python
python test.py -i path/to/training/dataset/pan16/english/ -m ./models/en.bin -o ./results  
```
Thus will test a pretrained model, provided by the -m flag, on a dataset, provided by the -i flag, and write the predictions about age-gender in the folder provided by -o flag.  It will also print accuracy and a cnofusion matrix per task, if true labels are availabel.


## Configuration:
Configuration follows the same conventions used for [PAN15 submission](https://github.com/iit-Demokritos/pangram).
In the config folder is a toy setup of the configuration for pangram. It is based on the
[YAML](http://yaml.org) format.

We use the [tictacs module](https://github.com/kbogas/tictacs) in order to create a modular-formalised workflow. It is mainly a wrapper around [sklearn's pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), that enables us to use config files.

Settings currently configurable are:
- Pan dataset settings for each language
- Feature groupings, preprocessing for each feature group, and classifier settings

In config/languages there is a file for each language which specifies where each attribute
to be predicted is in the truth file that contains the label for the training set. For each
of these attributes, you can set a file that contains the feature grouping and preprocessing
settings. In the example provided the mapping is the same for each language, but this need
not be the case.

In config/recipes the settings for each task can be found. The format is in the form:

    pipeline:
        label: english
        estimator: Pipeline
        estimator_pkg: sklearn.pipeline
        estimator_params:
            steps:
                - preprocess
                - label: features
                  estimator: FeatureUnion
                  estimator_pkg: sklearn.pipeline
                  estimator_params:
                    transformer_list:
                        - 3grams
                        - soac_model
                - svm
>
In the above snippet the *label* are identifiers that are expected to be found in the same .yml recipe file. They are unique for each element in the recipe.  

The *estimator* is the name of the function to be found in a package.  

The *estimatoor_pkg* is the name of the module where we can find the function

The *estimator_params* is a list of the parameters of the function *estimator*

E.g.: Concering for example the final svm classifier we can find in the rest of the file:

    svm:
      label: svm
      estimator: LinearSVC
      estimator_pkg: sklearn.svm
      estimator_params:
        C: 10
        class_weight: 'balanced'

## Final Results
- **1st place** in global ranking concering the **English** dataset
- Our team placed **6th** overall in global rankings. (22 teams in total) 

The final results regarding overall Average Accuracy:

| Team Name        | Global Score | Engilsh | Spanish | Dutch |
|-------------------------|--------|--------|--------|--------|
| Busger et al.           | 0.5258 | 0.3846 | 0.4286 | 0.4960 |
| Modaresi et al.         | 0.5247 | 0.3846 | 0.4286 | 0.5040 |
| Bilan et al.            | 0.4834 | 0.3333 | 0.3750 | 0.5500 |
| Modaresi(a)             | 0.4602 | 0.3205 | 0.3036 | 0.5000 |
| Markov et al.           | 0.4593 | 0.2949 | 0.3750 | 0.5100 |
| **Bougiatiotis & Krithara** | 0.4519 | **0.3974** | 0.2500 | 0.4160 |
| Dichiu & Rancea         | 0.4425 | 0.2692 | 0.3214 | 0.5260 |
| Devalkeneer             | 0.4369 | 0.3205 | 0.2857 | 0.5060 |
| Waser*                  | 0.4293 | 0.3205 | 0.2679 | 0.5320 |
| Bayot & Gonรงalves      | 0.4255 | 0.2179 | 0.3036 | 0.5680 |
| Gencheva et al.         | 0.4015 | 0.2564 | 0.2500 | 0.5100 |
| Deneva                  | 0.4014 | 0.2051 | 0.2679 | 0.6180 |

## License
Copyright 2016 NCSR Demokritos submission for Pan 2016, Konstantinos Bougiatiotis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

## Citation

I you want to cite us in your work, please use the following bibtex entry:
```
@inproceedings{BougiatiotisK16,
  author    = {Konstantinos Bougiatiotis and
               Anastasia Krithara},
  title     = {Author Profiling using Complementary Second Order Attributes and Stylometric
               Features},
  booktitle = {Working Notes of {CLEF} 2016 - Conference and Labs of the Evaluation
               forum, {\'{E}}vora, Portugal, 5-8 September, 2016.},
  pages     = {836--845},
  year      = {2016},
  crossref  = {DBLP:conf/clef/2016w},
  url       = {http://ceur-ws.org/Vol-1609/16090836.pdf},
  timestamp = {Thu, 11 Aug 2016 15:07:52 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/conf/clef/BougiatiotisK16}
}
```

