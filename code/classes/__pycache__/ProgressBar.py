# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:15:09 2022
@author: florent.brondolo
"""
import sys


class ProgressBar(object):
    """
   A class used to represent a progress bar in the terminal.
   Attributes
   ----------
   end : int
       The value at which the progress bar ends.
   start : int
       The value at which the progress bar starts.
   txt : str
       Optional text to display alongside the progress bar.
   _bar_length : int
       The length of the progress bar (number of characters).
   _level : int
       The current progress level.
   _ratio : float
       The ratio of current progress relative to the total progress range.
   _level_chars : int
       The number of characters representing the completed portion of the bar.
   _plotted : bool
       Boolean flag indicating whether the bar has been displayed at least once.
   """
    DEFAULT_BAR_LENGTH = 35
    DEFAULT_CHAR_ON = '■'
    DEFAULT_CHAR_OFF = '□'

    def __init__(self, end, start=0, txt=""):
        """
       Initializes the ProgressBar object with a start and end value, and optional text.
       Parameters
       ----------
       end : int
           The value at which the progress bar ends.
       start : int, optional
           The value at which the progress bar starts (default is 0).
       txt : str, optional            text to display alongside the progress bar (default is empty string).
       """
        self.end = end
        self.start = start
        self.txt = txt
        self._bar_length = self.__class__.DEFAULT_BAR_LENGTH
        self._level = start
        # Initialize _level here
        self._ratio = 0
        # Initialize _ratio here
        self._level_chars = 0
        # Initialize _level_chars here
        self._plotted = False
        self.set_level(self.start)

    def set_level(self, level):
        """
       Updates the current level of the progress bar and calculates related attributes.
       Parameters
       ----------
       level : int
           The new progress level to set. Will be clamped within the start and end bounds.
       """
        self._level = level
        if level < self.start:
            self._level = self.start
        if level > self.end:
            self._level = self.end
        self._ratio = float(self._level - self.start) / float(self.end - self.start)
        self._level_chars = int(self._ratio * self._bar_length)

    def plot_progress(self):
        """
       Displays the progress bar in the terminal with the current progress level.
       """
        sys.stdout.write("\r %3i%% %s%s %s" % (
            int(self._ratio * 100.0),
            self.__class__.DEFAULT_CHAR_ON * int(self._level_chars),
            self.__class__.DEFAULT_CHAR_OFF * int(self._bar_length - self._level_chars),
            '- [' + self.txt + ']',
        ))
        sys.stdout.flush()
        self._plotted = True

    def set_and_plot(self, level):
        """
       Updates the progress level and plots the progress bar if the level has changed.
       Parameters
       ----------
       level : int
           The new progress level to set and display.
       """
        old_chars = self._level_chars
        self.set_level(level)
        if not self._plotted or old_chars != self._level_chars:
            self.plot_progress()

    def __add__(self, other):
        """
       Adds a number to the current progress level and updates the progress bar.
       Parameters
       ----------
       other : float or int
           The value to add to the current progress level.
       Returns
       -------
       ProgressBar
           The updated ProgressBar object.
       """
        assert isinstance(other, (float, int)), "can only add a number"
        self.set_and_plot(self._level + other)
        return self

    def __sub__(self, other):
        """
       Subtracts a number from the current progress level and updates the progress bar.
       Parameters
       ----------
       other : float or int
           The value to subtract from the current progress level.
       Returns
       -------
       ProgressBar
           The updated ProgressBar object.
       """
        return self.__add__(-other)

    def __iadd__(self, other):
        """
       Adds a number to the current progress level in place.
       Parameters
       ----------
       other : float or int
           The value to add to the current progress level.
       Returns
       -------
       ProgressBar
           The updated ProgressBar object.
       """
        return self.__add__(other)

    def __isub__(self, other):
        """
       Subtracts a number from the current progress level in place.
       Parameters
       ----------
       other : float or int
           The value to subtract from the current progress level.
       Returns
       -------
       ProgressBar
           The updated ProgressBar object.
       """
        return self.__add__(-other)

    def __del__(self):
        """
       Cleans up when the ProgressBar object is deleted by printing a newline.
       """
        sys.stdout.write("\n")
