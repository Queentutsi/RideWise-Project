# RIDEWISE PROJECT

RideWise is a European mobility technology company operating in five major cities (London, Berlin, Amsterdam, Barcelona, Milan) and supporting over 200,000 customers and 800,000+ monthly trips. The business faces key challenges including high customer churn, limited behavioural insights, low promotional effectiveness, and no predictive analytics to identify at‑risk users.

This project builds a complete customer analytics (cleaning, EDA, feature engineering) and machine‑learning and modelling pipeline to help RideWise  improve retention, optimise marketing spend, and enable real‑time operational decision‑making. It includes churn prediction, customer segmentation, feature engineering, ML model interpretation, monitoring and API deployment, and dashboard reporting, using a synthetic but realistic dataset representing mobility usage patterns across Europe.


## Business Challenge 
-   25% quarterly churn among regular users
-   Limited understanding of customer behaviour and segments
-   Low promotional campaign uptake and unclear ROI
-   No predictive capability for identifying at‑risk customers.
-   Manual customer analysis taking weeks per campaign
-   Lack of real‑time insights for operations
-   No integrated analytics system for portfolio performance
-   Competitive pressure requiring data‑driven retention strategies

 

## Machine  Learning Model used. 
Models used 
- Random Forest as the main model, 
- Logistic Regression as the baseline.
Churn Prediction:  
- Primary Model:Random Forest  
- Baseline Model:Logistic Regression  

Customer Segmentation:  
- Primary Model:K‑Means Clustering  
- Optional:DBSCAN for anomaly detection  

Why these models? 
They balance **accuracy**, **interpretability**, **scalability**, and **business usability**, making them ideal for mobility analytics and customer‑retention systems.

# Recommended Models for RideWise

## Customer Churn Prediction (Classification Problem).
RideWise has:
- 25% quarterly churn  
- behavioural + transactional + promotional + external features  
- a need for interpretability (business stakeholders must understand why customers churn)

Model choice: 
Primary Model - Random Forest
Why Random Forest fits RideWise perfectly:
- Handles mixed data types (categorical and numerical)  
- Works well with behavioural features (RFM, usage patterns, cancellations)  
- Captures non‑linear relationships  
- Provides feature importance which is crucial for business explainability  
- This reduces noise and missing values.  
- Performs strongly even without heavy hyperparameter tuning  

### Secondary Model (for comparison): Logistic Regression  
Use it because:
- It’s highly interpretable  
- Great baseline  
- Helps validate Random Forest results  
- Useful for dashboards and business presentations  

## Customer Segmentation (Unsupervised Problem)
RideWise needs:
- Behavioural grouping  
- Promotion targeting  
- Usage‑pattern clustering  
- City‑level differentiation  

### Best Model: K‑Means Clustering
Why K‑Means is ideal:
- Works extremely well with behavioural metrics  
- Easy to interpret and explain to non‑technical stakeholders  
- Produces stable, meaningful customer groups  
- Perfect for marketing segmentation and retention strategies  
- Fast and scalable for 200k+ customers  

### 🟦 Secondary Model: DBSCAN (optional)
Use only if:
- You want to detect outliers  
- You have irregular clusters  
- You want to identify “rare behaviour” customers  

K‑Means should be the primary segmentation model.

## Feature Engineering Strategy
RideWise’s dataset supports:
- RFM analysis (Recency, Frequency, Monetary)  
- Temporal features (monthly trends, peak hours, weekend usage)  
- Trip‑based metrics (distance, duration, cancellations)  
- Promotion behaviour (usage, responsiveness)  
- External factors (weather, holidays, events)
These features work extremely well with Random Forest + K‑Means.

## Model Deployment
RideWise needs real‑time scoring → the best approach is:

# FastAPI + MLflow
- FastAPI for serving churn predictions  
- MLflow for tracking experiments and model versions  
- Streamlit for dashboards  



### My brief understanding of what ML FLOW IS AND HOW TO TRAIN A MODEL. WHY DO WE TRAIN MODELS 
- ML flow is a full AI engineering platform that supports modern AI systems like LLMs - large Language Models, AI agents, and classic ML models.
- ML flow is an open source platform (largest open source) used to manage ML - Machine Learning lifecycle, including expereimentation, reproducibility, deployment, and a central model registery. 
- If offers 4 componenets. - ML flow tracking (expereiment tracking), ML flow projects, ML Flow models, and Model Registery. 
- ML flow is an experiment tracking tool. It can be used to deploy your model. 
- Lots of platforms support the use of ML flow. Pytouch, tensorflow, Google cloud, AWS cloud, Docker, Fastai, XGboost, kubernetes, Azure Machine learning, keras, spark MLlib databricks.
- ML flow is my project’s “control centre” where I can log metrics (accuracy, recall, precision, AUC), log parameters (learning rate, model type, features used), save the models used, compare experiments (which model performed best,which parameters worked, which feature set was strongest), register the best model, and prepare it for deployment. It removes chaos and replaces it with structure. 
SOURCE: COPILOT. 
- MLflow give you the freedom to use multiple models (Logistic Regression, Random Forest, XGBoost etc.), multiple feature sets, multiple preprocessing pipelines and multiple hyperparameters. This helps to pick the best performing model. 
- MLflow enables you to export your trained model as a fully functional predictive service. It allows you to wrap/package your MLflow model, Dockerfile, prediction API (FastAPI or Flask), and all environment dependencies into a Docker container, ensuring scalable, portable, and production‑ready deployment.


LLMs stands for Large Language Models. It refers to AI models (like GPT‑style models) that are trained on massive amounts of text so they can understand language, generate responses, summarise information, write code, and more.

- Some of the benefits of ML flow tracking are Benefits:

Experiment Organization: Track and compare multiple model experiments
Metric Visualization: Built-in plots and charts for model performance
Artifact Storage: Store models, plots, and other files with each run
Collaboration: Share experiments and results across teams. 

MLflow is vendor-neutral; whether you're building AI agents, LLM applications, or ML models, it give access to MLflow's core capabilities — tracing, evaluation, experiment tracking, deployment, and more.

- SOURCE : https://mlflow.org/docs/latest/ml/
