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
- a need for **interpretability** (business stakeholders must understand *why* customers churn)

### Best Model: Random Forest
**Why Random Forest fits RideWise perfectly:**
- Handles mixed data types (categorical + numeric)  
- Works well with behavioural features (RFM, usage patterns, cancellations)  
- Captures non‑linear relationships  
- Provides feature importance → crucial for business explainability  
- Robust to noise and missing values  
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