# Results Folder Structure

This  note outlines the structure and contents of the Results folder, which includes  evaluation results and predictions related to the EDGE framework.
## Directory Overview


- **Evaluations**
- **Predictions**
- **Explanation Visualizations**



### Evaluation

Contains evaluation results for all explainers in individual file inside separate folders for each GNN model(RGCN and RGAT) in JSON format.


### Prediction

This folder contains predictions from all explainers, organized in separate subfolders for each GNN model in individual file inside the separate folders for  each dataset  stored in csv file. The CSV file contains the predicitons made by each explainers along with  entity id (IRI), ground truths as well as prediction made by GNN models.


### Explanation Visulaizations

This directory contains visualization images produced as explanations by various GNN explainers.