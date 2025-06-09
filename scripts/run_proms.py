import argparse
from data.fetch_credit import CreditFetcher


"""
Execute the script with the following command at the root directory of the project:
python model_test/<script_name>.py --name <dataset_name>
"""
def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='linear model test')
    parser.add_argument('--name', type=str, choices=["credit"], help='dataset_name')
    parser.add_argument('--grid_search', type=int, choices=[0, 1], help='is_fix')
    return parser.parse_args()

"""
Select the dataset based on the name
"""
def dataset_selecter(name: str):
    if name == "credit":    
        credit_fetcher = CreditFetcher()
        train_X, test_X, train_y, test_y = credit_fetcher.load_data(encoding='ohe')
    else:
        raise ValueError(f"Dataset {name} not found")   
    return train_X, test_X, train_y, test_y