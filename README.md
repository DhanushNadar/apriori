# Association Rule Mining Projects - Run Guide

This workspace contains separate question folders for:
- Real datasets (Q01 to Q10)
- Custom/generated datasets (GQ01 to GQ10)

Each folder has:
- app.py
- requirements.txt
- One CSV file

## 1) Prerequisites

- Python 3.9+ (recommended)
- pip

Check versions:

```powershell
python --version
pip --version
```

## 2) Install Dependencies

Run this once from the main folder:

```powershell
pip install pandas matplotlib mlxtend
```

Alternative (install from any question folder requirements):

```powershell
pip install -r Q01_Online_Retail_Analysis/requirements.txt
```

## 3) Important: Where to Run Commands

Run all commands from this root folder:

DMBI Datasets

Reason:
- A few scripts (Q02, Q03, Q04) use root-relative CSV paths.
- Running from root works for all scripts consistently.

## 4) Run Real Dataset Questions (Q01-Q10)

```powershell
python Q01_Online_Retail_Analysis/app.py
python Q02_E_Commerce_Transactions_Analysis/app.py
python Q03_Supermarket_Basket_Analysis/app.py
python Q04_Food_Delivery_Orders_Analysis/app.py
python Q05_Pharmacy_Sales_Analysis/app.py
python Q06_Book_Store_Purchase_Analysis/app.py
python Q07_Movie_Recommendation_Analysis/app.py
python Q08_Electronics_Store_Analysis/app.py
python Q09_Restaurant_Order_Analysis/app.py
python Q10_Fashion_Retail_Analysis/app.py
```

## 5) Run Custom Dataset Questions (GQ01-GQ10)

```powershell
python GQ01_Student_Course_Selection_Analysis/app.py
python GQ02_Online_Learning_Module_Analysis/app.py
python GQ03_Restaurant_Combo_Analysis/app.py
python GQ04_Gym_Activity_Pattern_Analysis/app.py
python GQ05_Mobile_App_Usage_Analysis/app.py
python GQ06_ELibrary_Book_Borrowing_Analysis/app.py
python GQ07_Supermarket_Simulation_Analysis/app.py
python GQ08_Travel_Package_Preference_Analysis/app.py
python GQ09_Smart_Home_Device_Usage_Analysis/app.py
python GQ10_Event_Participation_Analysis/app.py
```

## 6) What Output to Expect

For each script:
- Console output:
  - Top items/categories
  - Frequent itemsets
  - Association rules (support, confidence, lift)
  - Interpretation of strong rules
- Plots:
  - Bar chart (top products/items)
  - Scatter plot (rule distribution)

Close each plot window to allow script completion.

## 7) Quick Troubleshooting

If module not found:

```powershell
pip install pandas matplotlib mlxtend
```

If plots do not show:
- Ensure you are running in an environment with GUI support.
- In VS Code, run in terminal (not background task) and wait for the plot window.

If file not found error appears:
- Confirm you are executing from the root DMBI Datasets folder.
- Confirm CSV exists in the expected question folder.

## 8) Optional: Run One Script At A Time With Logs

```powershell
python -u GQ04_Gym_Activity_Pattern_Analysis/app.py
```

-u helps print output immediately.
