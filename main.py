from avalanche.evaluation.metrics import bwt_metrics, accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training import BaseStrategy
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from base.scenario import get_dataset_nc_scenario
from models.base import get_cl_model

dataset = 'splitcifar10'
model_name = 'vgg11'


tasks = get_dataset_nc_scenario(name=dataset,
                                n_tasks=5,
                                return_task_id=True)

model = get_cl_model(model_name=model_name,
                     input_shape=(3, 32, 32),
                     method_name='gem')


eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=False, experience=True,
                     stream=True),
    # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    # timing_metrics(epoch=True),
    # forgetting_metrics(experience=True, stream=True),
    # cpu_usage_metrics(experience=True),
    # confusion_matrix_metrics(num_classes=tasks.n_classes, save_image=False, stream=True),
    # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()],
    benchmark=tasks,
    strict_checks=True
)

parameters = model.parameters()
opt = Adam(parameters, lr=0.001)
criterion = CrossEntropyLoss()

strategy = BaseStrategy(model=model,
                        criterion=criterion,
                        optimizer=opt,
                        train_epochs=10,
                        train_mb_size=32,
                        evaluator=eval_plugin,
                        device='cuda:0')

results = []
for experience in tasks.train_stream:
    print('task')
    strategy.train(experiences=experience)

    results.append(strategy.eval(tasks.test_stream))
    # print(model)

for k, v in eval_plugin.get_last_metrics().items():
    print(k, v)
