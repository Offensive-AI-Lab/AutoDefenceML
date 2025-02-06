import os


def get_files_package_root() -> str:
    """Returns the root directory of your project."""
    return os.path.dirname(os.path.abspath(__file__))


def get_dataset_package_root() -> str:
    """Returns the root directory of your project."""
    # return "\\".join([os.path.dirname(os.path.abspath(__file__)),"dataset"])
    return os.path.dirname(os.path.abspath(__file__)) + r"/dataset"

def get_dataloader_package_root() -> str:
    """Returns the root directory of your project."""
    # return "\\".join([os.path.dirname(os.path.abspath(__file__)),"dataloader"])
    return os.path.dirname(os.path.abspath(__file__)) + r"/dataloader"