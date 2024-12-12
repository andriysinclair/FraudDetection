import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Plotter:
    def __init__(self, df):
        self.df = df
        plt.style.use("ggplot")

    def display_target(self, target="is_fraud"):
        # Calculating frequencies
        frequencies = self.df[target].value_counts()

        # Calculate percentages
        percentages = (frequencies / frequencies.sum()) * 100
        percentages = np.round(percentages, 2)

        # Combine into a DataFrame
        frequencies_table = pd.DataFrame(
            {
                "Response": frequencies.index,
                "Frequency": frequencies.values,
                "Percentage": percentages.values,
            }
        )

        display(frequencies_table)

    def plot_correlations(self, target="is_fraud"):
        corr_table = self.df.corr()[target].sort_values(ascending=False)

        # Create the correlation plot
        plt.figure(figsize=(12, 6))
        plt.bar(x=corr_table.index, height=corr_table.values)
        plt.title("Correlation Plot")
        plt.xlabel("Features")
        plt.ylabel("Correlation")
        plt.xticks(rotation=90)  # Rotate x-ticks
        # plt.tight_layout()
        plt.show()
