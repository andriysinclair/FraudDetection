import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Plotter:

    def __init__(self, df):
        self.df = df

    def display_target(self, target):
        # Calculating frequencies
        frequencies = self.df[target].value_counts()

        # Calculate percentages
        percentages = (frequencies / frequencies.sum()) * 100
        percentages = np.round(percentages, 2)

        # Combine into a DataFrame
        frequencies_table = pd.DataFrame({
            'Response': frequencies.index,
            'Frequency': frequencies.values,
            'Percentage': percentages.values
        })

        display(frequencies_table)

    def plot_correlations(self):
        corr_matrix = self.df.corr()

        # Create the correlation plot
        plt.figure(figsize=(8, 6))
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()  # Add color bar for scale
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title('Correlation Matrix Heatmap')
        plt.show()




    
