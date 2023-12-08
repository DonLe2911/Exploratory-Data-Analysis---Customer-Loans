import sqlalchemy
from sqlalchemy import create_engine
import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

def load_credentials():
    with open("EDA_CustomerLoans/credentials.yaml", "r") as file:
        credentials = yaml.safe_load(file)
    return credentials

class RDSDatabaseConnector:
    def __init__(self):
        self.credentials = load_credentials()
        self.engine = self.create_engine()

    def create_engine(self):
        db_url = f"postgresql://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}"
        engine = create_engine(db_url)
        return engine
    
    def extract_data_to_dataframe(self):
        table_name = 'loan_payments'
        query = f"SELECT * FROM {table_name}"
        dataframe = pd.read_sql(query, self.engine)
        return dataframe
        
    def save_to_csv(self, dataframe, local_path, filename):
        full_path = os.path.join(local_path, filename)
        dataframe.to_csv(full_path, index=False)

    def load_from_csv(self, local_path, filename):
        full_path = os.path.join(local_path, filename)
        dataframe = pd.read_csv(full_path)
        return dataframe

connector = RDSDatabaseConnector()
data = connector.extract_data_to_dataframe()
connector.save_to_csv(data, "EDA_CustomerLoans", "loan_payments_data.csv")
loaded_data = connector.load_from_csv("EDA_CustomerLoans", "loan_payments_data.csv")

#print(loaded_data['term'])
#print(loaded_data['loan_amount'])
#print(loaded_data['loan_status'].unique())


class Analysis:
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def calculate_recovery_percentage(self):
        total_funded = self.loan_data['funded_amount'].sum()
        total_recovered = self.loan_data['total_payment'].sum()
        recovery_percentage = (total_recovered / total_funded) * 100
        return recovery_percentage

    def calculate_payments_and_recovery_percentage(self):
        payments_due = []
        total_funded = self.loan_data['funded_amount'].sum()
        total_recovered = self.loan_data['total_payment'].sum()
        total_payments_due = 0  # Initialize total_payments_due outside the loop

        for _, row in self.loan_data.iterrows():
        # Convert 'issue_date', 'last_payment_date', and 'term' to datetime
            issue_date = pd.to_datetime(row['issue_date'])
            last_payment_date = pd.to_datetime(row['last_payment_date'])
                  
        # Handle different data types for 'term' column
        term = row['term']

        if pd.notna(term):
            if isinstance(term, (int, float)):
                term_in_months = int(term)
            else:
                term_in_months = int(''.join(filter(str.isdigit, str(term))))

            # Calculate the projected end date based on the issue date and term
            projected_end_date = issue_date + pd.DateOffset(months=term_in_months)

            # Calculate the number of months between 'last_payment_date' and the projected end date
            months_to_project = min(((projected_end_date - last_payment_date).days // 30), 6)

            # Calculate payments due for the next 'months_to_project' months
            payment_due = row['instalment'] * months_to_project
            payments_due.append(payment_due)

            # Update total_payments_due within the loop
            total_payments_due += payment_due

        # Calculate the percentage of payments due next 6 months and total recovered against total funded
        forcasted_percentage_recovered_after_6_months = ((total_payments_due + total_recovered) / total_funded) * 100

        return forcasted_percentage_recovered_after_6_months
    
    def calculate_loss_percentage(self):
        # Filter data for loans marked as Charged Off
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off']

        # Calculate the percentage of charged-off loans
        charged_off_percentage = (len(charged_off_loans) / len(self.loan_data)) * 100

        # Calculate the total amount paid towards charged-off loans
        total_amount_paid = charged_off_loans['total_payment'].sum()

        #calcualte total borrowed from charged off loans
        total_borrowed_from_charged_off = charged_off_loans['loan_amount'].sum()

        #calculate loss in revenue
        loss_in_revenue = total_borrowed_from_charged_off - total_amount_paid

        return charged_off_percentage, total_amount_paid, loss_in_revenue
    
    def visualize_loss_projection(self):
        # Filter data for loans marked as Charged Off
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off']

        # Sort charged-off loans by last payment date
        charged_off_loans = charged_off_loans.sort_values(by='last_payment_date')

        # Initialize variables
        cumulative_loss = 0
        cumulative_loss_list = []

        # Iterate through charged-off loans
        for _, row in charged_off_loans.iterrows():
            issue_date = pd.to_datetime(row['issue_date'])
            last_payment_date = pd.to_datetime(row['last_payment_date'])
            term = row['term']

            if pd.notna(term):
                if isinstance(term, (int, float)):
                    term_in_months = int(term)
                else:
                    term_in_months = int(''.join(filter(str.isdigit, str(term))))
            else:
                term_in_months = 36

            # Calculate the projected end date based on the issue date and term
            projected_end_date = issue_date + pd.DateOffset(months=term_in_months)

            # Calculate the number of months between 'last_payment_date' and the projected end date
            remaining_term = max(((projected_end_date - last_payment_date).days // 30), 0)

            # Calculate projected loss for the remaining term
            projected_loss = row['instalment'] * remaining_term

            # Update cumulative loss
            cumulative_loss += projected_loss
            cumulative_loss_list.append(cumulative_loss)

        # Visualize the cumulative loss over time
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_loss_list, marker='o')
        plt.title('Cumulative Projected Loss Over Remaining Term of Charged-Off Loans')
        plt.xlabel('Charged-Off Loans (Sorted by Last Payment Date)')
        plt.ylabel('Cumulative Projected Loss ($)')
        plt.grid(True)
        plt.show()


    def possible_loss(self):
    # Filter data for customers at risk of moving to charged off status
        customers_risk = self.loan_data[
            (self.loan_data['loan_status'] == 'Late (31-120 days)') |
            (self.loan_data['loan_status'] == 'Late (16-30 days)')
        ]

    # Calculate the total lost if customers at risk convert to charged off
        total_risk_lost = (customers_risk['loan_amount'] - customers_risk['total_payment']).sum()

    # Filter data for charged-off loans
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off']

    # Calculate the total lost for charged-off loans
        total_charged_off_lost = charged_off_loans['loan_amount'].sum() - charged_off_loans['total_payment'].sum()

    # Calculate the total loss if customers at risk convert to charged off plus total charged-off loss
        total_loss = total_risk_lost + total_charged_off_lost

    # Calculate the percentage of total loss against the total loan amount
        total_loan_amount = self.loan_data['loan_amount'].sum()
        percentage_of_total_loan = (total_loss / total_loan_amount) * 100

    # Calculate the percentage of all customers who would be considered a risk of moving to charged off status
        number_customers_risk = len(customers_risk)
        percentage_risk = number_customers_risk / len(self.loan_data) * 100

        return number_customers_risk, percentage_risk, total_risk_lost, total_charged_off_lost, total_loss, percentage_of_total_loan

class IndicatorsOfLoss:
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def charged_off_customers(self, columns_of_interest):
        # Filter data for loans marked as Charged Off
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off']

        # create subset specific columns for the charged_off_loans dataframe
        charged_off_customers = charged_off_loans[columns_of_interest]

        return charged_off_customers, charged_off_loans
    
    def customers_at_risk(self, columns_of_interest):
        # Filter data for customers at risk of moving to charged off status
        customers_risk = self.loan_data[
            (self.loan_data['loan_status'] == 'Late (31-120 days)') |
            (self.loan_data['loan_status'] == 'Late (16-30 days)')
        ]

        # Create a subset with specific columns for the customers_risk dataframe
        loans_risk = customers_risk[columns_of_interest]

        return loans_risk, customers_risk
    
    def print_count_for_columns(data_frame, label):
        # Calculate and print the count for each unique value in each column of the DataFrame
        print(f"\nCount of each input for {label} DataFrame:")
        for column in data_frame.columns:
            column_counts = data_frame[column].value_counts()
            print(f"\n{column}:\n{column_counts}")




# Instantiate the Analysis class with your DataFrame
analysis_instance = Analysis(loaded_data)

   
""" # Calculate the recovery percentage 
recovery_percentage = analysis_instance.calculate_recovery_percentage()
print(f"Percentage of loans recovered against total funding: {recovery_percentage:.2f}")

# Calculate the percentage of payments due next 6 months + total recovered against total funded
forcasted_percentage_recovered_after_6_months = analysis_instance.calculate_payments_and_recovery_percentage()
print(f"Percentage of Payments Due Next 6 Months + Total Recovered Against Total Funded: {forcasted_percentage_recovered_after_6_months:.2f}")

# Calculate the loss percentage and total amount paid towards charged-off loans
charged_off_percentage, total_amount_paid, loss_in_revenue = analysis_instance.calculate_loss_percentage()

# Print the results
print(f"Percentage of Charged-Off Loans: {charged_off_percentage:.2f}%")
print(f"Total Amount Paid Towards Charged-Off Loans: ${total_amount_paid:.2f}")
print(f"Loss in revenue: ${loss_in_revenue:.2f}")

# Visualize the cumulative projected loss over the remaining term of charged-off loans
analysis_instance.visualize_loss_projection()

number_customers_risk, percentage_risk, total_risk_lost, total_charged_off_lost, total_loss, percentage_of_total_loan = analysis_instance.possible_loss()

print(f"Number of customers at risk of moving to charged off: {number_customers_risk}")
print(f"Percentage of customers at risk: {percentage_risk:.2f}%")
print(f"Total lost if they converted to charged off: ${total_risk_lost:.2f}")
print(f"Total lost for charged-off loans: ${total_charged_off_lost:.2f}")
print(f"Total loss (risk + charged-off): ${total_loss:.2f}") """

## Instantiate the IndicatorsOfLoss class with your DataFrame
indicators_instance = IndicatorsOfLoss(loaded_data)

columns_to_print = ['grade', 'purpose', 'home_ownership']

# Call the method to get specified columns for charged-off loans
charged_off_customers, charged_off_loans = indicators_instance.charged_off_customers(columns_to_print)

# Call the method to get specified columns for customers at risk
customers_risk_subset, customers_risk = indicators_instance.customers_at_risk(columns_to_print)

""" # Print the resulting DataFrames
print("Charged Off Customers:")
print(charged_off_customers)

print("\nCustomers at Risk:")
print(customers_risk_subset) """

""" # Print the count of each input for each column in the DataFrames
IndicatorsOfLoss.print_count_for_columns(charged_off_customers, "Charged Off Customers")
IndicatorsOfLoss.print_count_for_columns(customers_risk_subset, "Customers at Risk") """


#INDICATORS OF LOSS SUMMARY
""" Some key trends summarised below:
- a large majority of charged off loans are grade C, B and D (71%)
- the biggest purpose by far for loans in charged off group is debt consolidation at 55%
- When look at ownership status for those in charged off group only 9% are home owners so we can see the biiger risk are those with rent or mortgage. 

If we then take a look at the at risk group we can predict which loans are at higher risk of changing to charged off by seeeing how many are:
- grade C, B or A AND
- purpose of debt consilidation AND
- have rent or mortgage status """

# Filter rows in customers_risk_subset based on conditions
""" filtered_rows = customers_risk_subset[
    (customers_risk_subset['grade'].isin(['C', 'B', 'D'])) &
    (customers_risk_subset['purpose'] == 'debt_consolidation') &
    (customers_risk_subset['home_ownership'].isin(['RENT', 'MORTGAGE']))
] """

# Count the number of rows in the filtered subset
#count_filtered_rows = len(filtered_rows)

# Print the result
#print(f"Number of customers at risk with grade C, B, or D, purpose of debt consolidation, and rent or mortgage ownership: {count_filtered_rows}")


#total = 294 out of 686 (43%) of the at risk group are at high risk of being converted to chargeed off
