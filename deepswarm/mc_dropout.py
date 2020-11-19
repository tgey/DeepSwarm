from typing import Dict, List
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .log import Log
import tqdm

def mc_dropout_class_prediction(model: tf.keras.Model, image: np.ndarray, label: int ,mc_samples: int) -> np.ndarray:
    Log.enable()
    image = image[np.newaxis]
    
    output_shape = model.layers[-1].output_shape
    prediction_list = np.zeros((mc_samples, output_shape[1]))

    # Perfom T stochastic foward passes 
    for i in tqdm.tqdm(range(mc_samples)):
        prediction = model.predict(image)
        prediction_list[i, :] = prediction
    accs = []
    for y_p in prediction_list:
        accs.append(1 if label == y_p.argmax() else 0)

    mc_acc = sum(accs)/len(accs)
    Log.info("MC accuracy: {:.1%}".format(mc_acc))
    
    return prediction_list, mc_acc

def mc_dropout_image_prediction(model, data_input ,mc_samples: int) -> dict:
    shape = np.shape(data_input)
    prediction_list = np.zeros((mc_samples, shape[1], shape[2], shape[3]))

    # Perfom T stochastic foward passes 
    for i in tqdm.tqdm(range(mc_samples)):
        prediction = model.predict(data_input)
        prediction_list[i, :, :, :] = prediction
    std = prediction_list.std(axis=0)
    mean = prediction_list.mean(axis=0)
    output = {
        'predictions': prediction_list,
        'std': std[:,:, 0],
        'mean': mean,
    }
    return output

def mc_dropout_cycle_prediction(prediction_fn: callable, model, input ,mc_samples: int) -> dict:
    shape = np.shape(input)
    prediction_list = np.zeros((mc_samples, shape[1], shape[2], shape[3]))

    # Perfom T stochastic foward passes 
    for i in tqdm.tqdm(range(mc_samples)):
        prediction = prediction_fn(model, input)
        prediction_list[i, :, :, :] = prediction
    std = prediction_list.std(axis=0)
    mean = prediction_list.mean(axis=0)
    output = {
        'predictions': prediction_list,
        'std': std,
        'mean': mean,
    }
    return output

def plot_uncertainty(data):
    predictions = data['mc_samples_predictions']

    sns.kdeplot(predictions[:, data['true_label']], shade=True)
    plt.xlim(0, 1)
    plt.xlabel('P(diseased | x)')
    plt.ylabel('density [a.u.]')
    plt.title('Predictive posterior ($\mu_{pred}=%.2f$;$\sigma_{pred}=%.2f$)' % (data['mu_pred_label'], data['std_pred_label']))
    plt.show()