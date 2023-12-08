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
#commenting out next two instantiations as dont need to repeat now that data is saved locally, means this script will run a lot quicker loading the data from local
#data = connector.extract_data_to_dataframe()
#connector.save_to_csv(data, "EDA_CustomerLoans", "loan_payments_data.csv")

loaded_data = connector.load_from_csv("EDA_CustomerLoans", "loan_payments_data.csv")

# Instantiate the Analysis class with your DataFrame
analysis_instance = Analysis(loaded_data)

# Calculate the recovery percentage 
recovery_percentage = analysis_instance.calculate_recovery_percentage()
print(f"Percentage of loans recovered against total funding: {recovery_percentage:.2f}%")

# Calculate the percentage of payments due next 6 months + total recovered against total funded
forcasted_percentage_recovered_after_6_months = analysis_instance.calculate_payments_and_recovery_percentage()
print(f"Percentage of Payments Due Next 6 Months + Total Recovered Against Total Funded: {forcasted_percentage_recovered_after_6_months:.2f}%")


pd.set_option('display.max_columns', None)

#print(loaded_data.dtypes)

class DataTransform:
    def convert_to_datetime(self, dataframe, column_names):
        for column_name in column_names:
            #convert selected columns to datetime
            dataframe[column_name] = pd.to_datetime(dataframe[column_name], format = "%b-%Y").dt.strftime("%m-%Y")

    def convert_to_category(self, dataframe, column_names):
        for column_name in column_names:
            #convert selected columns to category type
            dataframe[column_name] = dataframe[column_name].astype('category')

    def convert_to_numeric_months(self, dataframe, column_names, default_value=None):
        for column_name in column_names:
            # Extract numeric part from the string (e.g., '36 months' to 36)
            extracted_numeric = dataframe[column_name].str.extract('(\d+)')
            
            # Convert the extracted numeric part to numeric values
            dataframe[column_name + '_in_months'] = pd.to_numeric(extracted_numeric[0], errors='coerce')
            
            if default_value is not None:
                # Fill NaN values with the specified default value
                dataframe[column_name + '_in_months'].fillna(default_value, inplace=True)
    
    def convert_to_numeric_years(self, dataframe, column_names, default_value=None):
        for column_name in column_names:
            # Extract numeric part from the string (e.g., '6 years' to 6)
            extracted_numeric = dataframe[column_name].str.extract('(\d+)')
            
            # Convert the extracted numeric part to numeric values
            dataframe[column_name] = pd.to_numeric(extracted_numeric[0], errors='coerce')
            
            if default_value is not None:
                # Fill NaN values with the specified default value
                dataframe[column_name].fillna(default_value, inplace=True)

            # Rename the column
            dataframe.rename(columns={column_name: column_name + '_in_years'}, inplace=True)


#TRANSFORMING THE DATA --------------------------------------------------------------------------------------------------------------------------------
transformer = DataTransform()

# Convert multiple columns to datetime
date_columns = ['last_credit_pull_date', 'issue_date', 'last_payment_date', 'next_payment_date', 'earliest_credit_line']
transformer.convert_to_datetime(loaded_data, date_columns)

# Convert multiple columns to category
category_columns = ['home_ownership', 'verification_status', 'purpose', 'loan_status', 'payment_plan', 'application_type']
transformer.convert_to_category(loaded_data, category_columns)

#convert multuiple columns to int by extracting number
numerical_columns_months = ['term']
transformer.convert_to_numeric_months(loaded_data, numerical_columns_months, default_value=0)

#convert multuiple columns to int by extracting number
numerical_columns_years = ['employment_length']
transformer.convert_to_numeric_years(loaded_data, numerical_columns_years, default_value=0)

original_data = loaded_data
#print(original_data)
#--------------------------------------------------------------------------------------------------------------------------------------------------------

class DataFrameInfo:
    
    def __init__(self, loaded_data):
        self.loaded_data = loaded_data

    def print_data_types(self):
        #print data types
        print(self.loaded_data.dtypes)

    def extract_column_stats(self):
        column_stats = {}

        numeric_columns = self.loaded_data.select_dtypes(include='number').columns

        for column_name in numeric_columns:
            median_value = self.loaded_data[column_name].median()
            std_value = self.loaded_data[column_name].std()
            mean_value = self.loaded_data[column_name].mean()

            column_stats[column_name] = {
                'Median': median_value,
                'Standard Deviation': std_value,
                'Mean': mean_value
            }

        return column_stats

    def extract_dataframe_stats(self):
        numeric_columns = self.loaded_data.select_dtypes(include='number').columns
        median_values = self.loaded_data[numeric_columns].median()
        std_values = self.loaded_data[numeric_columns].std()
        mean_values = self.loaded_data[numeric_columns].mean()

        dataframe_stats = {
            'Median': median_values,
            'Standard Deviation': std_values,
            'Mean': mean_values
        }

        return dataframe_stats
    
    def count_distinct_values(self):
        categorical_columns = self.loaded_data.select_dtypes(include='category').columns
        distinct_values_count = {}

        for column_name in categorical_columns:
            distinct_values_count[column_name] = self.loaded_data[column_name].nunique()

        return distinct_values_count
    

    def identify_skewed_columns(self, skew_threshold=1.0):
        numeric_columns = self.loaded_data.select_dtypes(include='number').columns
        skewed_columns = []

        for column_name in numeric_columns:
            skewness = self.loaded_data[column_name].skew()
            if abs(skewness) > skew_threshold:
                skewed_columns.append(column_name)

        return skewed_columns
    
    def calculate_IQR_skew_values_fixed(self, skew_values_fixed):
        Q1 = skew_values_fixed.quantile(0.25)
        Q3 = skew_values_fixed.quantile(0.75)
        IQR = Q3 - Q1
        return IQR
    
    def calculate_correlation_matrix(self):
        correlation_matrix = self.loaded_data.corr()
        return correlation_matrix
    
    
#GENERAL USEFUL INFO FROM DATA-------------------------------------------------------------------------------------------------------------------------
#data_info = DataFrameInfo(loaded_data)

#Print data types
#data_info.print_data_types()

#Extract column stats
#column_stats = data_info.extract_column_stats()
#print("Column Stats:")
#print(column_stats)

#Extract dataframe stats
#df_stats = data_info.extract_dataframe_stats()
#print("Dataframe Stats:")
#print(df_stats)

#Count distinct values in categorical columns
#distinct_values_count = data_info.count_distinct_values()
#print("Distinct Values Count in Categorical Columns:")
#print(distinct_values_count)

#print("DataFrame Shape:", loaded_data.shape)

null_count = original_data.isnull().sum()
percentage_null = (original_data.isnull().mean()* 100) 

#Print the count of NULL values in each column
#print("Count of NULL values in each column:")
#print(null_count)
#Print the percentage of NULL values in each column
#print("\nPercentage of NULL values in each column:")
#print(percentage_null)

#-------------------------------------------------------------------------------------------------------------------------------------


class DataFrameTransform:
    def impute_null_values(self, dataframe):
        for column_name in dataframe.columns:
            if dataframe.dtypes[column_name] == 'float64' or dataframe.dtypes[column_name] == 'int64':
                # Impute with the median for numerical columns
                if dataframe[column_name].isnull().any():
                    median_value = dataframe[column_name].median()
                    dataframe[column_name].fillna(median_value, inplace=True)
            elif dataframe.dtypes[column_name] == 'object':
                # Impute with the most frequent value for categorical columns
                if dataframe[column_name].isnull().any():
                    most_frequent_value = dataframe[column_name].mode().iloc[0]
                    dataframe[column_name].fillna(most_frequent_value, inplace=True)
        return dataframe
    
    def identify_and_apply_best_transformation(self, dataframe, skewed_columns):
        best_transformation = {}
        best_reduction = {}

        for column_name in skewed_columns:
            # Identify the current skewness
            original_skewness = dataframe[column_name].skew()

            # Apply various transformations
            log_transformed = np.log1p(dataframe[column_name])
            sqrt_transformed = np.sqrt(dataframe[column_name])

            # Calculate skewness for each transformation
            log_skewness = log_transformed.skew()
            sqrt_skewness = sqrt_transformed.skew()

            # Find the transformation that results in the biggest reduction in skew
            if abs(log_skewness) < abs(sqrt_skewness):
                best_transformation[column_name] = 'log'
                best_reduction[column_name] = abs(original_skewness) - abs(log_skewness)
                dataframe[column_name] = log_transformed
            else:
                best_transformation[column_name] = 'sqrt'
                best_reduction[column_name] = abs(original_skewness) - abs(sqrt_skewness)
                dataframe[column_name] = sqrt_transformed

        return best_transformation, best_reduction, dataframe
    
    def remove_outliers_auto(self, dataframe):
        removed_outliers_indices = {}

        # Identify and remove outliers for all numeric columns
        for column_name in dataframe.select_dtypes(include='number').columns:
            # Calculate the Interquartile Range (IQR) for the column
            Q1 = dataframe[column_name].quantile(0.25)
            Q3 = dataframe[column_name].quantile(0.75)
            IQR = Q3 - Q1

            # Define the lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify and remove outliers
            outliers = (dataframe[column_name] < lower_bound) | (dataframe[column_name] > upper_bound)
            removed_outliers_indices[column_name] = dataframe[outliers].index

            # Remove outliers from the dataframe
            dataframe = dataframe[~outliers]

        # Return the dataframe without outliers and the indices of removed outliers
        return dataframe, removed_outliers_indices
    
    def remove_highly_correlated_columns(self, dataframe, correlation_threshold=0.8, exclude=[]):
        # Exclude non-numeric columns
        numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
        subset_dataframe = dataframe[numeric_columns]

        # Calculate the correlation matrix
        correlation_matrix = subset_dataframe.corr()

        # Find highly correlated columns
        highly_correlated_columns = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                    colname = correlation_matrix.columns[i]
                    highly_correlated_columns.add(colname)

        # Exclude specified columns from being dropped
        highly_correlated_columns = [col for col in highly_correlated_columns if col not in exclude]

        # Remove highly correlated columns
        dataframe_no_high_correlation = dataframe.drop(columns=highly_correlated_columns)
        return dataframe_no_high_correlation    

#DROP, IMPUTE, Transform-----------------------------------------------------------------------------------------------------------------------------------------------

#drop columns
cleaned_data = loaded_data.copy()
columns_to_drop = ['mths_since_last_delinq', 'mths_since_last_record', 'next_payment_date', 'mths_since_last_major_derog']
cleaned_data.drop(columns=columns_to_drop)

#print("Percentage of NULL values in each column after drop:")
#print(loaded_data.isnull().mean()* 100)

transformer = DataFrameTransform()

transformer.impute_null_values(cleaned_data)
#print(cleaned_data)
null_count = cleaned_data.isnull().sum()
#print(null_count)

#select sqr or log to fix skew
data_info = DataFrameInfo(cleaned_data)
skewed_columns = data_info.identify_skewed_columns(skew_threshold=1.0)
numeric_columns = cleaned_data.select_dtypes(include='number').columns
skew_values = cleaned_data[skewed_columns].skew()
#print("Skewness values for each column:")
#print(skew_values)

#print("Best Transformation for Each Skewed Column:")
#for column_name, transformation in best_transformation.items():
#    print(f"{column_name}: {transformation}")

#print("Reduction in Skewness for Each Skewed Column:")
#for column_name, reduction in best_reduction.items():
#    print(f"{column_name}: {reduction}")

#Apply and print the best transformations
best_transformation, best_reduction, dataframe_fixed = transformer.identify_and_apply_best_transformation(loaded_data, skewed_columns)
skew_values_fixed = dataframe_fixed[skewed_columns].skew()
#print(skew_values_fixed)

# Create an instance of DataFrameInfo
data_info = DataFrameInfo(cleaned_data)

# Call the method and print the result
IQR_skew_values_fixed = data_info.calculate_IQR_skew_values_fixed(skew_values_fixed)
#print("Interquartile Range (IQR) for skew_values_fixed:", IQR_skew_values_fixed)

# Call the remove_outliers_auto method
cleaned_data_without_outliers, removed_outliers_indices = transformer.remove_outliers_auto(cleaned_data)

# Print information about removed outliers
#for column_name, indices in removed_outliers_indices.items():
#    print(f"Removed outliers from {column_name}:")
#    print(cleaned_data.loc[indices])

# call the remove highly correlated columns method
exclude = ['funded_amount', 'funded_amount_inv', 'instalment']
dataframe_no_high_correlation = transformer.remove_highly_correlated_columns(dataframe_fixed, correlation_threshold=0.8, exclude=exclude)

#print(dataframe_no_high_correlation.columns)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Analysis:
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def calculate_recovery_percentage(self):
        total_funded = self.loan_data['funded_amount'].sum()
        total_recovered = self.loan_data['total_payment'].sum()
        print(total_funded)
        print(total_recovered)
        recovery_percentage = (total_recovered / total_funded) * 100

        return recovery_percentage
    
    def calculate_payments_and_recovery_percentage(self):
        payments_due = []
        total_funded = self.loan_data['funded_amount'].sum()
        total_recovered = self.loan_data['total_payment'].sum()

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
            else:
                # Set a default term value (you can adjust this as needed)
                term_in_months = 36

            # Calculate the projected end date based on the issue date and term
            projected_end_date = issue_date + pd.DateOffset(months=term_in_months)

            # Calculate the number of months between 'last_payment_date' and the projected end date
            months_to_project = min(((projected_end_date - last_payment_date).days // 30), 6)

            # Calculate payments due for the next 'months_to_project' months
            payment_due = row['instalment'] * months_to_project
            payments_due.append(payment_due)

        # Sum the payments due for all rows
        total_payments_due = np.sum(payments_due)

        # Calculate the percentage of payments due next 6 months and total recovered against total funded
        forcasted_percentage_recovered_after_6_months = ((total_payments_due + total_recovered) / total_funded) * 100

        return forcasted_percentage_recovered_after_6_months

# Instantiate the Analysis class with your DataFrame
analysis_instance = Analysis(loaded_data)

# Calculate the recovery percentage 
recovery_percentage = analysis_instance.calculate_recovery_percentage()
print(f"Percentage of loans recovered against total funding: {recovery_percentage:.2f}%")

# Calculate the percentage of payments due next 6 months + total recovered against total funded
forcasted_percentage_recovered_after_6_months = analysis_instance.calculate_payments_and_recovery_percentage()
print(f"Percentage of Payments Due Next 6 Months + Total Recovered Against Total Funded: {forcasted_percentage_recovered_after_6_months:.2f}%")







class Plotter:
    def visualize_null_removal(self, original_data, cleaned_data):
        # Check if there are any NULL values in either original or cleaned data
        if original_data.isnull().any().any() or cleaned_data.isnull().any().any():
            # Get the columns with NULL values in the original data
            null_columns_original = original_data.columns[original_data.isnull().any()]
            
            # Get the percentage of NULL values for each column in the original data
            null_percentage_original = original_data[null_columns_original].isnull().mean() * 100

            # Get the columns with NULL values in the cleaned data
            null_columns_cleaned = cleaned_data.columns[cleaned_data.isnull().any()]

            # Get the percentage of NULL values for each column in the cleaned data
            null_percentage_cleaned = cleaned_data[null_columns_cleaned].isnull().mean() * 100

            # Plot only columns with NULL values
            plt.figure(figsize=(10, 6))
            plt.bar(null_percentage_original.index, null_percentage_original, color='blue', alpha=0.7, label='Original Data')
            plt.bar(null_percentage_cleaned.index, null_percentage_cleaned, color='red', alpha=0.7, label='Cleaned Data')

            plt.xlabel('Columns')
            plt.ylabel('Percentage of NULL values')
            plt.title('Comparison of NULL Values in Original and Cleaned Data')
            plt.xticks(rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()

            plt.show()
        else:
            print("No NULL values to visualize.")


    def visualize_scatter_null_removal(self, original_data, cleaned_data):
        if original_data.isnull().any().any() or cleaned_data.isnull().any().any():
                     
            null_columns_original = original_data.columns[original_data.isnull().any()]     
            null_percentage_original = original_data[null_columns_original].isnull().mean() * 100

            null_columns_cleaned = cleaned_data.columns[cleaned_data.isnull().any()]
            null_percentage_cleaned = cleaned_data[null_columns_cleaned].isnull().mean() * 100

            # Ensure both arrays have the same length by taking the union of columns
            columns_union = set(null_percentage_original.index).union(null_percentage_cleaned.index)
            
            null_percentage_original = null_percentage_original.reindex(columns_union).fillna(0)
            null_percentage_cleaned = null_percentage_cleaned.reindex(columns_union).fillna(0)

            plt.figure(figsize=(12, 8))

            x_values = np.arange(len(columns_union))

            plt.plot(x_values, null_percentage_original, marker='o', linestyle='-', label='Original Data', alpha=0.7)
            plt.plot(x_values, null_percentage_cleaned, marker='o', linestyle='-', label='Cleaned Data', alpha=0.7)

            plt.xlabel('Columns')
            plt.ylabel('Percentage of NULL values')
            plt.title('Comparison of NULL Values in Original and Cleaned Data')
            plt.xticks(x_values, columns_union, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No NULL values to visualize.")

    def visualize_reduction_in_skewness(self, original_data, transformed_data, skewed_columns):
        if not skewed_columns:
            print("No skewed columns to visualize.")
            return

        plt.figure(figsize=(12, 8))

        x_values = np.arange(len(skewed_columns))

        # Identify and remove outliers in both original and transformed data
        transformer = DataFrameTransform()
        original_data_no_outliers, _ = transformer.remove_outliers_auto(original_data)
        transformed_data_no_outliers, _ = transformer.remove_outliers_auto(transformed_data)

        original_skewness_no_outliers = original_data_no_outliers[skewed_columns].skew()
        transformed_skewness_no_outliers = transformed_data_no_outliers[skewed_columns].skew()

        plt.plot(x_values, original_skewness_no_outliers, marker='o', linestyle='-', label='Original Skewness (No Outliers)')
        plt.plot(x_values, transformed_skewness_no_outliers, marker='o', linestyle='-', label='Transformed Skewness (No Outliers)')

        plt.xlabel('Columns')
        plt.ylabel('Skewness Values')
        plt.title('Reduction in Skewness Visualization (Outliers Removed)')
        plt.xticks(x_values, skewed_columns, rotation=45, ha='right')
        plt.legend()
        plt.ylim(bottom=-5, top=5)
        plt.tight_layout()
        plt.show()

    def visualize_correlation_matrix(self, dataframe):
        # Exclude non-numeric columns from the correlation matrix
        numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = dataframe[numeric_columns].corr()

        # Plot the correlation matrix heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()


#Printing the graphs

#plotter = Plotter()
#plotter.visualize_scatter_null_removal(original_data, cleaned_data)            

#plotter = Plotter()
#plotter.visualize_null_removal(original_data, cleaned_data)

#plotter = Plotter()
#plotter.visualize_reduction_in_skewness(cleaned_data, dataframe_fixed, skewed_columns)

#plotter = Plotter()
#plotter.visualize_correlation_matrix(dataframe_fixed)

#plotter = Plotter()
#plotter.visualize_correlation_matrix(dataframe_no_high_correlation)

#print(dataframe_no_high_correlation.columns)