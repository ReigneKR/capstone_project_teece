# Capstone Project: Lifestyle and Learning
<b>Submitted By:</b> John Louis D. Lagramada, Reigne Kenneth R. Reyes, Nacer L. Lerit <br>
<b>Email:</b> johnlouis.dante@gmail.com <br>
<b>Data Storytelling:</b> TODO: add video link <br>
<b>Whitepaper:</b> TODO: add whitepaper relative link

## Project Summary
This section discusses the summary of the entire project from preparation of data to deriving actionable insights for improving student performance. In data storytelling, we framed the narrative of the data in a fictional school called Bagong Silang Intergalactic High School.

### Research Problem
This project aims to answer this following research problem using machine learning:

> How do lifestyle habits—such as study time, diet quality, and exercise frequency—influence examination performance of students?

We used unsupervised learning (K-Means) clustering and supervised learning (regression and classification algorithms) to answer this. 

### Approach

We first explored the data with exploratory data analysis through data visualizations. We determined the prevalence of outliers, handled missing data, preprocessed the data, and prepared it for machine learning using data transformation techniques such as standardization in a pipeline. We also engineered new features that is used by the model.

We then used the data provided in several machine learning algorithms. Initially, the data is clustered using <b>K-means</b> where we determined the appropriate number of clusters using both <b>Elbow Method</b> and <b>Silhouette Score</b>. With this, we were able to create per cluster models that can be used for <b>Mixture of Experts (MoE) approach</b>. We benchmarked the results for the per cluster models and one global model. In visualizing the K-means clusters, we used <b>principal component analysis</b> (PCA).

In training the supervised learning models, we used an <b>90-10</b> dataset split ratio. Moreover, we used <b>K-fold cross validation</b> for hyperparameter tuning and a <b>hold-out test set</b> that is purely isolated from training. The regression models we used are:
- linear regression
- decision tree regressor
- random forest regressor.

These models are evaluated with:
- R2 score
- mean absolute error
- root mean square error

For classification models, we used:
- logistic regression
- support vector classifier
- random forest classifier

These models are evaluated with:
- area under the receiver operating characteristic curve (AUC ROC)
- confusion matrix metrics
    - accuracy
    - precision
    - recall
    - F1 score

### Results

TODO: add results

### Conclusion

TODO: add conclusion

## Machine Learning Methods
This section discusses the technical approach used to solve the underlying research problem from exploratory data analysis to model evaluation and benchmarking. We show specific values used in training the model and decisions that we made along the way in squeezing out performance out of the models.

### Data Understanding and Preprocessing
TODO
### Exploratory Data Analysis
TODO
### Clustering
TODO
### Regression
TODO
### Classification
TODO 
## Key Findings
This section discusses feature importance, cluster profiling, and actionable insights derived from the models used in the project.
### Feature Importance
TODO
### Cluster Profiling
TODO
### Best Models
TODO
### Actionable Insights
TODO
## Reproducibility
Developing the project requires certain versions of libraries. To ensure reproducibility, make sure to install all the modules as listed in the `requirements.txt`. The kernel of your Jupyter Notebook should use the virtual environment where the requirements are installed (this is implicit if the steps below are followed). For deterministic numerical outputs, a global variable seed is used which is 121.

1. Clone the repository.
    <pre><code>git clone https://github.com/lukasdante/capstone_teece2.git</code></pre>
2. Create a virtual environment.
    <pre><code>python -m venv env</code></pre>
3. Source your virtual environment.
    - In Linux/macOS
    <pre><code>source env/bin/activate</code></pre>
    - In Windows command prompt
    <pre><code>myenv\Scripts\activate.bat</code></pre>
    - In Windows PowerShell
    <pre><code>.\myenv\Scripts\activate.bat</code></pre>
    If policy error shows up, run the code below:
    <pre><code>Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser</pre></code>
4. Install libraries.
    <pre><code>pip install -r requirements.txt</code></pre>
5. Open Jupyter notebook.
    <pre><code>jupyter notebook</code></pre>
6. Navigate the `source/` directory to view the code. The directory tree is as follows:
    <pre><code>TODO: Add directory tree by tree -L 1.</code></pre>

TODO: pip freeze > requirements.txt

## License
This project is licensed under the [MIT License](./LICENSE). Feel free to use, modify, and distribute this project with proper attribution.