### **Week 1: Command Line & Environment Mastery**

**Goal**: Master CLI for ML workflows, automate tasks, and manage environments.

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **File System Navigation**  <br>\- Practice `ls -l`, `cd ~`, `tree`  <br>\- Organize ML folders (e.g., `data/raw`, `models/`) | **Task**: Write a script to auto-generate ML project folders (e.g., `mkdir -p data/{raw,processed}`). | [Linux Command Line Basics (Coursera)](https://www.coursera.org/learn/unix) |
| **Day 2** | **File Manipulation**  <br>\- Batch rename files with `rename`  <br>\- Use `find` to locate files | **Project**: Create a script to clean temporary files (e.g., `*.log`, `*.tmp`). | [The Linux Command Line (Book)](https://linuxcommand.org/tlcl.php) |
| **Day 3** | **Text Processing**  <br>\- Extract specific columns from CSV using `awk`  <br>\- Filter logs with `grep` | **Task**: Analyze server logs to find error patterns (use `grep -C 5 "ERROR"`). | [Text Processing with Unix (FreeCodeCamp)](https://www.freecodecamp.org/news/) |
| **Day 4** | **Environment Setup**  <br>\- Compare `conda` vs `venv`  <br>\- Install CUDA for GPU support | **Project**: Create a `requirements.txt` for a ML project (include `numpy>=1.21`). | [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html) |
| **Day 5** | **SSH & Remote Work**  <br>\- Set up SSH keys  <br>\- Use `tmux` for session management | **Task**: Deploy a Python script to AWS EC2 and run it remotely. | [DigitalOcean SSH Tutorial](https://www.digitalocean.com/community/tutorials/ssh-essentials) |
| **Day 6** | **Automation**  <br>\- Write a cron job to backup datasets daily  <br>\- Use `rsync` for incremental backups | **Project**: Automate dataset downloads using `wget` and `cron`. | [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) |
| **Day 7** | **Review & Debugging**  <br>\- Debug a broken shell script  <br>\- Optimize a slow pipeline | **Project**: Build a CLI tool to preprocess CSV files (e.g., remove nulls, add headers). | [ShellCheck (Linter)](https://www.shellcheck.net/) |

* * *

### **Week 2: Python Programming & Data Structures**

**Goal**: Strengthen Python skills for ML with real-world datasets and algorithms.

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **Python Basics**  <br>\- Solve 5 LeetCode Easy problems (e.g., Two Sum)  <br>\- Use `typing` for type hints | **Task**: Build a CLI-based quiz game with user input validation. | [Python for Everybody (Coursera)](https://www.coursera.org/specializations/python) |
| **Day 2** | **Data Structures**  <br>\- Implement a linked list  <br>\- Use `defaultdict` for counting | **Project**: Create a cache system using `lru_cache` decorator. | [Real Python Data Structures](https://realpython.com/python-data-structures/) |
| **Day 3** | **OOP for ML**  <br>\- Design a `Dataset` class with `load()`, `preprocess()` methods | **Task**: Build a class to handle CSV/JSON data loading and basic stats. | [Python OOP Tutorial](https://realpython.com/python3-object-oriented-programming/) |
| **Day 4** | **Advanced Functions**  <br>\- Use `*args`/`**kwargs`  <br>\- Write a decorator to log function execution time | **Project**: Create a retry decorator for API calls. | [Fluent Python (Book)](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) |
| **Day 5** | **Error Handling**  <br>\- Custom exceptions for data validation  <br>\- Use `pytest` for unit tests | **Task**: Write tests for your `Dataset` class (e.g., test missing file handling). | [Python Testing with pytest (Book)](https://pythontest.com/pytest-book/) |
| **Day 6** | **Python for Data**  <br>\- Vectorize operations with NumPy  <br>\- Optimize loops with `numba` | **Project**: Speed up a Pandas operation using vectorization (e.g., `df.apply()` → `np.where()`). | [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) |
| **Day 7** | **Mini Project**  <br>\- Build a CLI tool to fetch live stock prices (APIs + `argparse`). | **Advanced**: Add caching with `@lru_cache` and error handling for invalid symbols. | [Alpha Vantage API](https://www.alphavantage.co/) |

* * *

### **Week 3: Data Analysis & Visualization**

**Goal**: Master Pandas, EDA, and storytelling with data.

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **Pandas Basics**  <br>\- Clean the [Titanic dataset](https://www.kaggle.com/c/titanic)  <br>\- Use `query()` for filtering | **Task**: Calculate survival rates by gender/class and visualize with bar plots. | [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas) |
| **Day 2** | **Data Cleaning**  <br>\- Handle missing data with `fillna()`  <br>\- Detect outliers with `z-score` | **Project**: Clean the [Housing Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). | [Data Cleaning Checklist](https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4) |
| **Day 3** | **EDA**  <br>\- Use `seaborn` for pair plots  <br>\- Analyze correlations with `.corr()` | **Task**: Identify trends in [Netflix Movies](https://www.kaggle.com/shivamb/netflix-shows) (e.g., genre popularity over time). | [Python Data Science Handbook (EDA)](https://jakevdp.github.io/PythonDataScienceHandbook/) |
| **Day 4** | **Advanced Pandas**  <br>\- Reshape data with `melt()`/`pivot_table()`  <br>\- Merge datasets with `merge()` | **Project**: Combine COVID-19 case data with vaccination rates for a country. | [Pandas Merging Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html) |
| **Day 5** | **Time Series**  <br>\- Resample stock data to weekly intervals  <br>\- Plot rolling averages | **Task**: Forecast Bitcoin prices using moving averages (use `yfinance` API). | [Time Series Analysis Guide](https://machinelearningmastery.com/time-series-datasets-for-ml/) |
| **Day 6** | **Storytelling**  <br>\- Create a Jupyter Notebook report with Plotly visualizations | **Project**: Build an interactive dashboard for Airbnb price analysis. | [Plotly Tutorial](https://plotly.com/python/) |
| **Day 7** | **Mini Project**  <br>\- Analyze [Spotify Tracks](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-600k-tracks) and predict song popularity. | **Advanced**: Deploy the analysis as a Streamlit app. | [Streamlit Docs](https://docs.streamlit.io/) |

* * *

### **Key Improvements Over Original Roadmap**

1.  **More Projects**: Added 2-3 projects/week (CLI tools, data pipelines, APIs).
    
2.  **Real-World Data**: Integrated Kaggle datasets and APIs (Alpha Vantage, Spotify).
    
3.  **Advanced Python**: Focus on OOP, decorators, and performance optimization.
    
4.  **Tooling**: Emphasis on `pytest`, `numba`, `Streamlit`, and cloud deployment.
    
5.  **Coursera Integration**: Added relevant courses for structured learning.
    
6.  **Frontend Synergy**: Projects like Streamlit dashboards leverage your frontend skills.
    

### **Week 4: Machine Learning Fundamentals**

**Goal**: Master supervised learning, model evaluation, and scikit-learn workflows.

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **Supervised Learning Basics**  <br>\- Solve 5 classification problems on [Kaggle Learn](https://www.kaggle.com/learn/intro-to-machine-learning)  <br>\- Use `train_test_split` | **Task**: Predict Titanic survival using logistic regression. | [Coursera: Supervised Machine Learning](https://www.coursera.org/learn/machine-learning) |
| **Day 2** | **Regression Models**  <br>\- Implement linear regression with `sklearn`  <br>\- Use `PolynomialFeatures` for non-linear data | **Project**: Predict bike rentals using [UCI Bike Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). | [Scikit-Learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) |
| **Day 3** | **Classification Models**  <br>\- Train a decision tree on [Iris Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)  <br>\- Visualize tree with `plot_tree` | **Task**: Build a flower species classifier with 95%+ accuracy. | [Decision Tree Visualization Guide](https://mljar.com/blog/visualize-decision-tree/) |
| **Day 4** | **Model Evaluation**  <br>\- Plot ROC curves with `roc_curve`  <br>\- Calculate precision/recall tradeoffs | **Project**: Evaluate a spam classifier using [SMS Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). | [Model Evaluation Metrics Guide](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) |
| **Day 5** | **Cross-Validation**  <br>\- Use `KFold` and `cross_val_score`  <br>\- Compare stratified vs regular splits | **Task**: Validate your bike rental model with 5-fold CV. | [Scikit-Learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html) |
| **Day 6** | **Feature Engineering**  <br>\- Encode categorical variables with `OneHotEncoder`  <br>\- Scale features with `StandardScaler` | **Project**: Preprocess the [Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). | [Feature Engineering for ML](https://www.coursera.org/learn/feature-engineering) |
| **Day 7** | **Mini Project**  <br>\- Build a [Streamlit](https://streamlit.io/) app to predict house prices. Include sliders for user input. | **Advanced**: Deploy the app on Streamlit Cloud. | [Streamlit Tutorial](https://docs.streamlit.io/) |

* * *

### **Week 5: Advanced ML & Hyperparameter Tuning**

**Goal**: Optimize models with hyperparameter tuning, pipelines, and ensemble methods.

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **Hyperparameter Tuning**  <br>\- Use `GridSearchCV` to optimize SVM parameters  <br>\- Compare `RandomizedSearchCV` | **Task**: Tune a random forest model on the [Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). | [Hyperparameter Tuning Guide](https://www.coursera.org/learn/deep-neural-network/lecture/RgKox/hyperparameter-tuning) |
| **Day 2** | **ML Pipelines**  <br>\- Build a pipeline with `StandardScaler` and `SVC`  <br>\- Save pipelines with `joblib` | **Project**: Create a pipeline for [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). | [Scikit-Learn Pipelines](https://scikit-learn.org/stable/modules/compose.html) |
| **Day 3** | **Ensemble Learning**  <br>\- Implement bagging with `RandomForestClassifier`  <br>\- Compare AdaBoost vs Gradient Boosting | **Task**: Boost accuracy on the [MNIST Dataset](https://www.kaggle.com/c/digit-recognizer) using ensembles. | [Ensemble Methods Explained](https://www.coursera.org/lecture/competitive-data-science/ensemble-methods-1WIRx) |
| **Day 4** | **Unsupervised Learning**  <br>\- Cluster [Mall Customer Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) with K-Means  <br>\- Visualize clusters with PCA | **Project**: Segment customers into 3-5 groups based on spending habits. | [Coursera: Unsupervised Learning](https://www.coursera.org/learn/machine-learning-unsupervised-learning) |
| **Day 5** | **Dimensionality Reduction**  <br>\- Reduce features in [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) using PCA  <br>\- Compare t-SNE vs UMAP | **Task**: Visualize high-dimensional data in 2D. | [PCA vs t-SNE Guide](https://towardsdatascience.com/pca-vs-tsne-elizabeth-5b9e4b6a4a3d) |
| **Day 6** | **Model Interpretability**  <br>\- Use SHAP values to explain model predictions  <br>\- Plot feature importance | **Project**: Explain why your credit card fraud model flags specific transactions. | [SHAP Tutorial](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%2520introduction%2520to%2520explainable%2520AI%2520with%2520Shapley%2520values.html) |
| **Day 7** | **Mini Project**  <br>\- Build a Gradio app to classify handwritten digits. Let users draw digits and see predictions. | **Advanced**: Host the app on Hugging Face Spaces. | [Gradio Docs](https://gradio.app/docs/) |

* * *

### **Week 6: Neural Networks & Deep Learning Basics**

**Goal**: Dive into neural networks with TensorFlow/Keras and build your first CNN.

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **Neural Network Basics**  <br>\- Build a DNN with Keras `Sequential` API  <br>\- Train on [Fashion MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) | **Task**: Achieve 85%+ accuracy on Fashion MNIST. | [Coursera: Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) |
| **Day 2** | **Optimization Techniques**  <br>\- Experiment with SGD, Adam, and RMSprop  <br>\- Use learning rate schedulers | **Project**: Compare training curves for different optimizers. | [Keras Optimizers Guide](https://keras.io/api/optimizers/) |
| **Day 3** | **CNNs for Image Data**  <br>\- Build a CNN with `Conv2D` and `MaxPooling` layers  <br>\- Visualize filters with `Keract` | **Task**: Classify [Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats) images. | [CNN Explainer (Interactive Tool)](https://poloclub.github.io/cnn-explainer/) |
| **Day 4** | **Transfer Learning**  <br>\- Fine-tune `VGG16` on a custom dataset  <br>\- Use data augmentation with `ImageDataGenerator` | **Project**: Build a food classifier using [Food-101 Dataset](https://www.kaggle.com/datasets/kmader/food41). | [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning) |
| **Day 5** | **Overfitting Solutions**  <br>\- Add dropout and L2 regularization  <br>\- Use early stopping with `EarlyStopping` | **Task**: Reduce overfitting in your Cats vs Dogs model. | [Keras Callbacks Docs](https://keras.io/api/callbacks/) |
| **Day 6** | **Model Deployment**  <br>\- Convert a Keras model to TensorFlow Lite  <br>\- Deploy to Android with [TF Lite Demo App](https://www.tensorflow.org/lite/examples) | **Project**: Build a mobile app to classify flowers from camera input. | [TensorFlow Lite Tutorial](https://www.tensorflow.org/lite/guide) |
| **Day 7** | **Mini Project**  <br>\- Create a React.js frontend + FastAPI backend for your Cats vs Dogs model. Let users upload images. | **Advanced**: Use `Docker` to containerize the app. | [FastAPI + React Tutorial](https://testdriven.io/blog/fastapi-react/) |

* * *

### **Key Enhancements**

1.  **Frontend Integration**: Projects like Gradio/Streamlit apps and React+FastAPI deployments leverage your frontend skills.
    
2.  **Coursera Alignment**: Added courses like *Neural Networks and Deep Learning* and *Hyperparameter Tuning*.
    
3.  **Real-World Tools**: SHAP, TensorFlow Lite, Docker, and Hugging Face Spaces.
    
4.  **Hands-On Datasets**: Fashion MNIST, Credit Card Fraud, Cats vs Dogs, and Food-101.
    
5.  **Deployment Focus**: Mobile apps, cloud deployment, and web apps to build a portfolio.
    

### **Week 7: Natural Language Processing (NLP)**

**Goal**: Master text processing, transformer models, and building NLP pipelines.

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **Text Preprocessing**  <br>\- Tokenize text with `spaCy`  <br>\- Remove stopwords and lemmatize | **Task**: Clean and preprocess [Twitter Sentiment Data](https://www.kaggle.com/datasets/kazanova/sentiment140). | [spaCy Tutorial](https://spacy.io/usage) |
| **Day 2** | **Vectorization**  <br>\- Use `TF-IDF` and `CountVectorizer`  <br>\- Compare with `Word2Vec` embeddings | **Project**: Build a news topic classifier using [BBC News Dataset](https://www.kaggle.com/datasets/pariza/bbc-news-summary). | [Gensim Word2Vec Guide](https://radimrehurek.com/gensim/models/word2vec.html) |
| **Day 3** | **Transformer Models**  <br>\- Fine-tune `BERT` for sentiment analysis  <br>\- Use `Hugging Face` pipelines | **Task**: Classify movie reviews using [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). | [Hugging Face Course](https://huggingface.co/course/chapter1) |
| **Day 4** | **Named Entity Recognition (NER)**  <br>\- Train a custom NER model with `spaCy`  <br>\- Visualize entities with `displacy` | **Project**: Extract company names and dates from financial news articles. | [spaCy NER Tutorial](https://spacy.io/usage/linguistic-features#named-entities) |
| **Day 5** | **Text Generation**  <br>\- Use `GPT-2` to generate text  <br>\- Experiment with temperature and top-k sampling | **Task**: Build a Shakespeare-style poetry generator. | [Transformers Text Generation](https://huggingface.co/docs/transformers/tasks/text_generation) |
| **Day 6** | **Deploy NLP Models**  <br>\- Build a FastAPI endpoint for text classification  <br>\- Containerize with Docker | **Project**: Deploy a sentiment analysis API and integrate it with a React.js frontend. | [FastAPI + React Tutorial](https://testdriven.io/blog/fastapi-react/) |
| **Day 7** | **Mini Project**  <br>\- Create a Streamlit chatbot using `ChatGPT` or `Llama 2`.  <br>\- Host it on Hugging Face Spaces. | **Advanced**: Add voice input/output using `SpeechRecognition` and `gTTS`. | [Streamlit Chat Tutorial](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps) |

* * *

### **Week 8: Advanced Deep Learning**

**Goal**: Dive into RNNs, GANs, and reinforcement learning (RL).

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **RNNs & LSTMs**  <br>\- Predict stock prices with LSTM  <br>\- Use `TensorFlow`’s `TimeseriesGenerator` | **Task**: Forecast COVID-19 cases using [JHU Dataset](https://github.com/CSSEGISandData/COVID-19). | [TensorFlow Time Series Guide](https://www.tensorflow.org/tutorials/structured_data/time_series) |
| **Day 2** | **Attention Mechanisms**  <br>\- Implement a transformer from scratch  <br>\- Compare with `TensorFlow`’s `MultiHeadAttention` | **Project**: Build a language translation model (English → Spanish). | [Transformer Paper](https://arxiv.org/abs/1706.03762) |
| **Day 3** | **Generative Models (GANs)**  <br>\- Train a GAN to generate MNIST digits  <br>\- Visualize training with `TensorBoard` | **Task**: Generate synthetic faces using [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). | [GAN Tutorial](https://www.tensorflow.org/tutorials/generative/dcgan) |
| **Day 4** | **Reinforcement Learning (RL)**  <br>\- Train an agent to play `CartPole` with `OpenAI Gym`  <br>\- Use Q-learning | **Project**: Solve the `MountainCar` problem using Deep Q-Networks (DQN). | [OpenAI Gym Docs](https://www.gymlibrary.dev/) |
| **Day 5** | **Model Optimization**  <br>\- Prune a CNN with `TensorFlow Model Optimization`  <br>\- Quantize to INT8 with `TFLite` | **Task**: Optimize your Cats vs Dogs model for mobile deployment. | [TF Model Optimization](https://www.tensorflow.org/model_optimization) |
| **Day 6** | **Edge AI**  <br>\- Deploy a model on Raspberry Pi using `TF Lite`  <br>\- Use `OpenCV` for real-time inference | **Project**: Build a plant disease detector with a Pi camera. | [TensorFlow Lite Raspberry Pi Guide](https://www.tensorflow.org/lite/guide/python) |
| **Day 7** | **Mini Project**  <br>\- Build a recommendation system using collaborative filtering (e.g., MovieLens dataset). | **Advanced**: Deploy it as a React.js app with personalized suggestions. | [MovieLens Dataset](https://grouplens.org/datasets/movielens/) |

* * *

### **Week 9: MLOps & Production Systems**

**Goal**: Master CI/CD, model monitoring, and scalable deployment.

| Day | Topics & Exercises | Projects & Advanced Tasks | Resources & Tools |
| --- | --- | --- | --- |
| **Day 1** | **CI/CD for ML**  <br>\- Set up GitHub Actions to retrain models on new data  <br>\- Add automated testing | **Task**: Automate model retraining for your house price prediction API. | [GitHub Actions for ML](https://docs.github.com/en/actions) |
| **Day 2** | **Model Serving**  <br>\- Serve models with `TensorFlow Serving`  <br>\- Benchmark performance with `Locust` | **Project**: Deploy a ResNet model and stress-test it with 100+ RPS. | [TF Serving Guide](https://www.tensorflow.org/tfx/guide/serving) |
| **Day 3** | **Feature Stores**  <br>\- Set up a feature store with `Feast`  <br>\- Integrate with ML pipelines | **Task**: Build a fraud detection pipeline using real-time features. | [Feast Documentation](https://docs.feast.dev/) |
| **Day 4** | **Model Monitoring**  <br>\- Track drift with `Evidently`  <br>\- Set up alerts for degraded performance | **Project**: Monitor your credit card fraud model in production. | [Evidently AI](https://www.evidentlyai.com/) |
| **Day 5** | **Kubernetes Scaling**  <br>\- Deploy a model with `Kubernetes`  <br>\- Use `HorizontalPodAutoscaler` | **Task**: Scale your sentiment analysis API to handle traffic spikes. | [Kubernetes ML Deployment](https://www.youtube.com/watch?v=6BYq6hNhceI) |
| **Day 6** | **Security & Compliance**  <br>\- Encrypt model artifacts with `AWS KMS`  <br>\- Implement RBAC with `Auth0` | **Project**: Secure your FastAPI endpoints with OAuth2 and JWT tokens. | [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/) |
| **Day 7** | **Mini Project**  <br>\- Build an end-to-end ML pipeline: Data ingestion → Training → Deployment → Monitoring. | **Advanced**: Use `Airflow` or `Prefect` for workflow orchestration. | [Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html) |

* * *

### **Key Enhancements**

1.  **Frontend Integration**: React.js apps, Streamlit chatbots, and Raspberry Pi projects.
    
2.  **Industry Tools**: Feast, Evidently, TF Serving, and Kubernetes.
    
3.  **Real-World Projects**: Fraud detection, recommendation systems, and edge AI.
    
4.  **Coursera Alignment**: Include courses like [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) and [MLOps Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops).
    
5.  **Portfolio Boost**: Deploy 5+ projects (APIs, web apps, edge devices) to showcase full-stack ML skills.
    

* * *

### **Recommended Books**

- 📘 [**Natural Language Processing in Action**](https://www.manning.com/books/natural-language-processing-in-action)
    
- 📘 [**Deep Learning for Coders**](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527)
    
- 📘 [**Machine Learning Engineering**](https://www.amazon.com/Machine-Learning-Engineering-Production-Machine/dp/1098117969)