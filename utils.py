import os


def get_save_path(scenario_name: str,
                  plugin: str,
                  plugin_name: str,
                  model_name: str,
                  exp_n: int = None):

    base_path = os.getcwd()
    if exp_n is None:
        return os.path.join(base_path,
                            model_name,
                            scenario_name,
                            plugin,
                            plugin_name)

    experiment_path = os.path.join(base_path,
                                   model_name,
                                   scenario_name,
                                   plugin,
                                   plugin_name,
                                   f'exp_{exp_n}')
    return experiment_path
