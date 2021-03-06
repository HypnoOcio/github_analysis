# GitHub repository analysis
This work aims to analyse information about repositories in order to determine
the quality of the repository and its lifespan.

# Structure of project
1. src  - directory containing all source codes used during implementation
    - big_query			- the directory with queries from BigQuery
    - data				- the directory with information about repositories
	- model_weights 	- the directory with pretrained weights
	- models 			- the directory with model architectures
	- training_utils 	- the directory with scripts needed for training
	- utils_notebooks 	- extract data from GithubAPI
	- training_models.ipynb 	- training and validating
	- dataset_preprocess.ipynb	- steps to preprocess dataset
	- notebooks_from_github.ipynb 	- access to GithubAPI
2. text - the thesis text directory
	- fig 		- the directory with figures
	- ref.bib 	- the bibliography resource
	- thesis.pdf 	- the thesis text in PDF format
	- thesis.tex 	- the thesis text in LATEX format
