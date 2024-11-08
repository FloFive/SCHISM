# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:55:14 2022

@author: florent.brondolo
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


class LRFind(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, n_rounds):
        """
            Initializes the learning rate scheduler.

            Args:
                min_lr (float): Minimum learning rate.
                max_lr (float): Maximum learning rate.
                n_rounds (int): Number of rounds for the learning rate schedule.

            Attributes:
                step_up (float): Factor to increase the learning rate each round.
                lrs (list): List to store learning rates.
                losses (list): List to store losses.
                weights (None): Placeholder for model weights.
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_up = (max_lr / min_lr) ** (1 / n_rounds)
        self.lrs = []
        self.losses = []
        self.weights = None

    def on_train_begin(self):
        """
            Prepares the model for training by setting initial weights and learning rate.

            Sets:
                weights: Initializes model weights.
                model.optimizer.lr: Sets the learning rate to the minimum value.
        """
        self.weights = self.model.get_weights()
        self.model.optimizer.lr = self.min_lr

    def on_train_batch_end(self, logs=None):
        """
            Updates learning rate and records metrics at the end of each training batch.

            Args:
                logs (dict, optional): Contains batch loss information.

            Updates:
                lrs (list): Appends current learning rate.
                losses (list): Appends current batch loss.
                model.optimizer.lr: Increases learning rate by step_up factor.
                model.stop_training: Stops training if learning rate exceeds max_lr.
        """
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs["loss"])
        self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True

    def on_train_end(self):
        """
            Restores model weights at the end of training.
        """
        self.model.set_weights(self.weights)
