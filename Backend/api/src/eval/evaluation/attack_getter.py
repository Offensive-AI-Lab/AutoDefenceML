from importlib import import_module

def get_attacks(paths_list):
    """
    :param paths_list: list of attack paths
    :return: list of attack objects
    """
    attacks_list = []
    for path in paths_list:
        split_idx = path.rfind(".")
        art_path = path[:split_idx]
        attack = path[split_idx + 1:]
        attacks_list.append(getattr(import_module(art_path), attack))
    return attacks_list



