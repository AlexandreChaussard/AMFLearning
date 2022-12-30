from river import ensemble
from river import evaluate
from river import metrics
from river.datasets import Bananas

dataset = Bananas().take(500)

model = ensemble.AMFClassifier(n_classes=2, n_estimators=10, use_aggregation=True, dirichlet=0.1)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
