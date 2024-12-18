import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Plotter:
    """

    Allows different plots to be carried out on dataframe

    """

    def __init__(self, df):
        """



        Args:
            df (pd.DataFrame): dateframe on which to carry out plotting
        """
        self.df = df
        plt.style.use("ggplot")

    def display_target(self, target="is_fraud"):
        """display_target

        displays the proportion of fraudulent and non-fraudulent transactions

        Args:
            target (str, optional): target column. Defaults to "is_fraud".
        """
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

    def plot_correlations(self, Plots_folder, file_name, target="is_fraud"):
        """plot_correlations

        Plots correlation betweeen features and target column

        Args:
            Plots_folder (str): path to folder where to save plots
            file_name (str): desired file name (without suffix i.e. .pdf)
            target (str, optional): target column. Defaults to "is_fraud".
        """
        corr_table = self.df.corr()[target].sort_values(ascending=False)

        # Create the correlation plot
        plt.figure(figsize=(12, 6))
        plt.bar(x=corr_table.index, height=corr_table.values)
        plt.title("Correlation Plot")
        plt.xlabel("Features")
        plt.ylabel("Correlation")
        plt.xticks(rotation=90)  # Rotate x-ticks
        plt.tight_layout()
        plt.savefig(Plots_folder + "/" + file_name + ".pdf", format="pdf")
        plt.show()

    def bar_plot(
        self,
        feature_of_interest,
        top_n,
        Plots_folder,
        file_name,
        target="is_fraud",
        title=None,
        xlabel=None,
        ylabel=None,
    ):
        """bar_plot

        bar plots

        Args:
            feature_of_interest (str): column name for which to find distribution of fraudulent and non-fraudulent transactions (should be categorical)
            top_n (int): bar plots for top n categories
            Plots_folder (str): path to folder where to save plots
            file_name (str): file name without suffix
            target (str, optional): target column. Defaults to "is_fraud".
        """
        key_values = (
            self.df.groupby(feature_of_interest)[target]
            .sum()
            .sort_values(ascending=False)
        )
        x = key_values.index[:top_n]
        y = key_values.values[:top_n]
        # print(f"x: {x}")
        # print(f"y {y}")

        # Create bar plot
        plt.figure(figsize=(12, 6))
        plt.bar(x, y)
        plt.title(
            title if title else f"Most fraudulent transactions by {feature_of_interest}"
        )
        plt.xlabel(xlabel if xlabel else f"{feature_of_interest}")
        plt.ylabel(ylabel if ylabel else "Count")
        plt.xticks(rotation=90)  # Rotate x-ticks
        plt.tight_layout()
        plt.savefig(Plots_folder + "/" + file_name + ".pdf", format="pdf")
        plt.show()

    def box_plot(self, feature_of_interest, Plots_folder, file_name, target="is_fraud"):
        """box_plot

        box plots

        Args:
            feature_of_interest (str): Column for which to do box plot (should be numerical)
            Plots_folder (str): path to folder where to save plots
            file_name (str): file name without suffix
            target (str, optional): target column. Defaults to "is_fraud".
        """

        # Prepare data for the box plot
        fraudulent = self.df[self.df[target] == 1][feature_of_interest]
        non_fraudulent = self.df[self.df[target] == 0][feature_of_interest]

        # Create box plot
        plt.figure(figsize=(10, 6))
        plt.boxplot(
            [non_fraudulent, fraudulent],
            labels=["Non-Fraudulent", "Fraudulent"],
            patch_artist=True,  # Add color fill
        )

        # Customize the plot
        plt.title("Transaction Amount Distribution by Fraudulent Status")
        plt.xlabel("Transaction Type")
        plt.ylabel("Transaction Amount")
        # plt.yscale("log")  # Use log scale for better visualization if amounts vary significantly
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        plt.savefig(Plots_folder + "/" + file_name + ".pdf", format="pdf")
        plt.show()
