import pandas as pd


class DataLoader:
    def __init__(self, path, batch_size=128):
        self.path = path
        self.current_index = 0
        self._load_data()
        self.batch_size = batch_size

        # Identify the target column
        if 'target' in self.data.columns:
            self.target_column = 'target'
        elif 'label' in self.data.columns:
            self.target_column = 'label'
        else:
            self.target_column = self.data.columns[-1]

        self.data_loaded = False

    def _load_data(self):
        # Add logic to load data based on the path type
        if self.path.endswith('.csv'):
            self.data = pd.read_csv(self.path)
        elif self.path.endswith('.xlsx'):
            self.data = pd.read_excel(self.path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < len(self.data):
            start_index = self.current_index
            end_index = self.current_index + self.batch_size
            batch_data = self.data.iloc[start_index:end_index]
            self.current_index = end_index

            X = batch_data.drop(columns=[self.target_column])
            Y = batch_data[[self.target_column]]

            # Ensuring X and Y are DataFrames
            assert isinstance(X, pd.DataFrame), "X is not a pandas DataFrame"
            assert isinstance(Y, pd.DataFrame), "Y is not a pandas DataFrame"

            return X, Y
        else:
            self.current_index = 0
            raise StopIteration

    # YOU MUST IMPLEMENT THIS FUNCTION
    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.current_index < len(self.data):
            start_index = self.current_index
            end_index = self.current_index + batch_size
            batch_data = self.data.iloc[start_index:end_index]
            self.current_index = end_index

            X = batch_data.drop(columns=[self.target_column])
            Y = batch_data[[self.target_column]]

            # Ensuring X and Y are DataFrames
            assert isinstance(X, pd.DataFrame), "X is not a pandas DataFrame"
            assert isinstance(Y, pd.DataFrame), "Y is not a pandas DataFrame"

            return X, Y  # YOU MUST RETURN PANDAS DATAFRAMES
        else:
            self.current_index = 0
            raise StopIteration

    #DON'T MODIFY THIS FUNCTION
    def get_data(self):
        return self.get_batch(batch_size= len(self.data))
# Example usage:
# dataloader = DataLoader(r'C:\Users\user\PycharmProjects\api\src\bias_eval\FairGAN\default of credit card clients.csv')
# for X, Y in dataloader:
#     print(X)
#     print(Y)
#     target_name = Y.columns[0]
#     print(f"Target column: {target_name}")
#     dataframe = pd.concat([X, Y], axis=1)
#     print(dataframe.head())

