# Project Description

This project is an interactive web application built using Flask that embeds Tableau dashboards to visualize electric vehicle charging patterns, battery performance, and driving range analysis.

The main objective of this project is to understand the relationship between battery capacity, charging behavior, and vehicle efficiency using interactive dashboards.

# EV-Battery-Performance-and-Range-Monitoring-System
Accurate electric vehicle range analysis is crucial for optimizing trip planning, reducing range anxiety, and improving energy efficiency. By leveraging data analytics, this project identifies key factors affecting range, predicts achievable distances, and provides actionable insights to support drivers,and charging infrastructure planning.


# Technologies Used

Python
Flask
Tableau
HTML
CSS
Bootstrap
Pandas

# Dashboard link
https://public.tableau.com/views/EVBatteryPerformanceandRangeMonitoringSystem/EvDashBoard?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

# story link
https://public.tableau.com/views/Tableau_17718657991890/StoryofElectriccarsinIndia?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

# key Questions Explored

1. Which EV brands lead battery performance globally?
2. How does temperature affect EV range predictions?
Which Indian states dominate EV registrations?
What velocity optimizes battery energy efficiency?
Which models show fastest range degradation?
How do charging cycles impact battery SoH?
What drives India's EV adoption growth rate?
Which factors cause real vs. lab range gaps?
Top 5 EV manufacturers by market share?
Optimal charging station locations in India?


# EV Battery & Range Insights
Battery Electric Vehicles (BEVs) dominate at 77.6% of registrations versus 22.4% Plug-in Hybrids (PHEVs), with average electric range reaching 67.86 miles due to battery tech advances. Tesla leads manufacturers, followed by Nissan, Chevrolet, Ford, and BMW, powering top models like Model Y, Model 3, and Nissan Leaf. Machine learning ensembles predict optimal velocity with 77-91% accuracy (R²), minimizing range errors to ~5-9 km via RF+ET+LSTM models.

# India EV Adoption Story
Registrations exploded 10x from 2020-2024, led by Uttar Pradesh, Maharashtra, Karnataka, and Tamil Nadu; Northeastern states show highest YoY growth. Electric four-wheelers hit 115,800 units in FY24-25 (15% YoY growth), e-rickshaws at 474,503, and e-carts at 65,060. Dashboards highlight brand-model counts, pricing variations, and global comparisons for market penetration trends.


# Recommended ML Algorithms

## Random Forest + Extra Trees + LSTM Ensemble
Purpose: Predict optimal driving velocity profiles achieving 77-91% accuracy (R²), minimizing range prediction errors to 5-9 km.
​
Why: RF/ET handle non-linear feature interactions (temperature, terrain, speed); LSTM captures temporal battery degradation patterns.

## Linear Regression
Purpose: Real-time State of Charge (SoC) prediction during charge-discharge cycles (99% R² score).
​
Why: Fast baseline for BMS integration, interpretable coefficients link voltage/capacity to SoH.




