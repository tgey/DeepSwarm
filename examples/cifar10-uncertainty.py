# Copyright (c) 2019 Edvinas Byla
# Licensed under MIT License

import context
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

import os, sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import traceback

from deepswarm.backends import Dataset, TFKerasBackend
from deepswarm.deepswarm import DeepSwarm
from deepswarm.mc_dropout import mc_dropout_class_prediction
from deepswarm.log import Log
Log.enable()

# Load CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# Create dataset object, which controls all the data
dataset = Dataset(
    training_examples=x_train,
    training_labels=y_train,
    testing_examples=x_test,
    testing_labels=y_test,
    validation_split=0.1,
)
# Create backend responsible for training & validating
backend = TFKerasBackend(dataset=dataset)
# Create DeepSwarm object responsible for optimization
deepswarm = DeepSwarm(backend=backend)
# Find the topology for a given dataset
try:
    topology = deepswarm.find_topology()
except:
    Log.error(f'{sys.exc_info()} occured')
    Log.error(f'{traceback.format_exc()}')
# Evaluate discovered topology
deepswarm.evaluate_topology(topology)
# Train topology on augmented data for additional 50 epochs
trained_topology = deepswarm.train_topology(topology, 50, augment={
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
})
# Evaluate the final topology
deepswarm.evaluate_topology(trained_topology)

# Evaluate model uncertainty using 10 test examples. Model should not be deterministic, otherwise, uncertainty would be null. 
# Here we kept Dropout layers activated during inference (set during model init).
model = trained_topology
overall_prediction_list = []
for X, y_true in zip(x_test[:10], y_test[:10]):
    label = np.argmax(y_true)

    # Predict 100 stochastic runs
    pred, mc_acc = mc_dropout_class_prediction(model=model, image=X, label=label, mc_samples=100)

    # Get mean and variance (uncertainty estimator).
    pred_coeff = pred.mean(axis=0) # Pred_coeff = np.array(nb_classes) mean values
    pred_coeff_std = pred.std(axis=0) # Pred_coeff = np.array(nb_classes) standard deviation values
    
    max_elem = np.amax(pred_coeff)
    max_elem_idx = np.argmax(pred_coeff)
    if max_elem_idx == label:
        Log.info(f'[Correct prediction]\tPred Label {max_elem_idx} with mu:{max_elem} -  std:{pred_coeff_std[max_elem_idx]}')       
    else:
        Log.error(f'[Incorrect prediction]\tPred Label {max_elem_idx} with mu:{max_elem} - std:{pred[:, max_elem_idx].std()}\n\t\t\tTrue label {label} with mu:{pred[:, label].mean()} - std:{pred[:, label].std()}')
    
    # Useful for > 2 classification models, otherwise, if the model is false, the true value is implicit.
    prediction = {
        'true_label': label,
        'mu_true_label': pred_coeff[label],
        'std_true_label': pred_coeff_std[label],
        'Pred_label': max_elem_idx,
        'mu_pred_label': pred_coeff[max_elem_idx],
        'std_pred_label': pred_coeff_std[max_elem_idx],
        'mc_samples_predictions': pred,
        'mc_accuracy': mc_acc,
        'prediction_coeff_mu': pred_coeff,
        'prediction_coeff_std': pred_coeff_std,
        'image': X,
    }
    overall_prediction_list.append(prediction)

    # Plot class ditribution, with which classes the model hesitated for this example.
    fig, _ = plt.subplots(6, 2, figsize=(12,12))
    for i, (mu, std, ax) in enumerate(zip(pred_coeff, pred_coeff_std, fig.get_axes())):
        sns.kdeplot(ax=ax, data=pred[:, i], shade=True)
        ax.hist(pred[:,i], bins=100, range=(0,1))
        ax.set_ylim([0, 100])
        ax.set_title("class: {} - $\mu: {:.1%}$ -  $\sigma: {:.2%}$".format(i, mu, std))
        ax.label_outer()
    fig.get_axes()[10].imshow(prediction['image'])
    plt.show()

# Overall performances
avg_coeff = np.array([i['prediction_coeff_mu'] for i in overall_prediction_list]).mean(axis=0)
classes = sorted([i for i in enumerate(avg_coeff)], key=lambda i:i[1])
over_mu = np.array([i['mu_pred_label'] for i in overall_prediction_list]).mean()
over_std = np.array([i['std_pred_label'] for i in overall_prediction_list]).mean()

#Ordered list by uncertainty estimation value
prediction_list_by_uncertainty = sorted(overall_prediction_list, key=lambda k: k['mu_true_label'], reverse=True)
plt.imshow(prediction_list_by_uncertainty[0]['image']) #easiest
plt.show()
plt.imshow(prediction_list_by_uncertainty[-1]['image']) #hardest
plt.show()
Log.warning(f'Overall mu {over_mu} and std {over_std}')