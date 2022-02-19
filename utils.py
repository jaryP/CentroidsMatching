from typing import Optional, Any

import avalanche


def get_tasks(name: str,
              tasks: int,
              shuffle: bool = True,
              train_transform: Optional[Any] = None,
              eval_transform: Optional[Any] = None,
              seed: Optional[int] = None):

    name = name.lower()

    for v in dir(avalanche.benchmarks):
        if v.lower() == name:
            return getattr(avalanche.benchmarks, v)(n_experiences=tasks,
                                                    return_task_id=True,
                                                    shuffle=shuffle,
                                                    train_transform=train_transform,
                                                    eval_transform=eval_transform,
                                                    seed=seed)


get_tasks('splitmnist', 1)
