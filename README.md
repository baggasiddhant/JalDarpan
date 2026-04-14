# JALDARPAN

This project started as a water demand forecasting system, but a big part of the work ended up being around handling and working with structured data properly.

It focuses on predicting water usage using time-series data, while also dealing with how that data is stored, processed, and queried efficiently.

---

## Project Overview

The goal was to predict water demand using historical and environmental data. But while building it, I had to deal with real issues like cleaning messy datasets, structuring data properly, and making sure it could be queried and processed without slowing things down.

So it’s not just a model project, it’s also about handling data in a practical way.

---

## What I worked on (relevant parts)

- Worked with 5+ years of time-series data (water usage, rainfall, etc.)
- Cleaned and structured datasets for consistent usage
- Built preprocessing pipelines for handling missing and inconsistent data
- Used SQL + Python to explore and analyze structured data
- Designed data flow for training and prediction
- Focused on keeping data processing efficient as dataset size increased

---

## Data Handling

The data wasn’t clean or uniform, so a lot of effort went into fixing that.

### Data used:
- Water consumption records  
- Weather data (rainfall, humidity, temperature)  
- Seasonal trends  
- Some region-level data  

### What I did:
- Handled missing values (basic imputation)
- Normalized data before feeding it into the model
- Created features like seasonal patterns and demand ratios
- Converted raw data into time-series sequences

This part actually took more time than the model itself.

---

## Model (brief)

Used an LSTM-based model for forecasting.

- 2 LSTM layers (100 units each)  
- Adam optimizer  
- MSE loss  
- 80/20 train-test split  

The model performed decently and predictions were close to actual trends.

---

## Backend / System Side

- Built a simple backend using Flask  
- Handled data input and prediction requests  
- Connected model outputs to a dashboard  
- Made sure data flows from input → processing → prediction worked reliably  

---

## What I learned 

This project made me realize:

- Data quality matters more than the model  
- Structuring data properly makes everything easier later  
- Querying and processing large datasets can become slow if not handled well  
- Even small inefficiencies add up when data grows  

It got me more interested in databases, query performance, and how data systems are designed in real applications.

---

## Future Work

- Move data handling to a proper database system (PostgreSQL)
- Work on better query performance for larger datasets  
- Try handling real-time data instead of static datasets  
- Explore scaling this beyond a single-machine setup  

---

## Tech Used

Python, SQL, Pandas, NumPy, TensorFlow/Keras, Flask, Plotly
