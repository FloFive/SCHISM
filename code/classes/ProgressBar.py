# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:15:09 2022
@author: florent.brondolo
"""
import sys

class ProgressBar(object):
   """
   A class to display a progress bar in the console.
   Attributes:
   ----------
   DEFAULT_BAR_LENGTH : int
       The default length of the progress bar in characters.
   DEFAULT_CHAR_ON : str
       The character used to represent the completed portion of the bar.
   DEFAULT_CHAR_OFF : str
       The character used to represent the remaining portion of the bar.
   """
   DEFAULT_BAR_LENGTH = 35
   # DEFAULT_CHAR_ON = '■█'□■
   DEFAULT_CHAR_ON = '■'
   DEFAULT_CHAR_OFF = '□'
   def __init__(self, end, start=0, txt=""):
       """
       Initialize a ProgressBar object.
       Parameters:
       ----------
       end : int
           The final value of the progress bar (100% completion).
       start : int, optional
           The initial value of the progress bar (default is 0).
       txt : str, optional
           An optional text to display beside the progress bar (default is an empty string).
       """
       self._levelChars = None
       self._ratio = None
       self._level = None
       self.end = end
       self.start = start
       self.txt = txt
       self._barLength = self.__class__.DEFAULT_BAR_LENGTH
       self.setLevel(self.start)
       self._plotted = False
   def setLevel(self, level):
       """
       Set the current progress level and update the progress bar accordingly.
       Parameters:
       ----------
       level : int
           The current value of the progress (between `start` and `end`).
       """
       self._level = level
       if level < self.start:
           self._level = self.start
       if level > self.end:
           self._level = self.end
       self._ratio = float(self._level - self.start) / float(self.end - self.start)
       self._levelChars = int(self._ratio * self._barLength)
   def plotProgress(self):
       """
       Plot the progress bar in the console, showing the current progress as a percentage.
       """
       sys.stdout.write("\r %3i%% %s%s %s" % (
           int(self._ratio * 100.0),
           self.__class__.DEFAULT_CHAR_ON * int(self._levelChars),
           self.__class__.DEFAULT_CHAR_OFF * int(self._barLength - self._levelChars),
           '- [' + self.txt + ']',
       ))
       sys.stdout.flush()
       self._plotted = True
   def setAndPlot(self, level):
       """
       Set the progress level and immediately plot the updated progress bar.
       Parameters:
       ----------
       level : int
           The new progress level.
       """
       oldChars = self._levelChars
       self.setLevel(level)
       if (not self._plotted) or (oldChars != self._levelChars):
           self.plotProgress()
   def __add__(self, other):
       """
       Increment the progress bar by a specified amount.
       Parameters:
       ----------
       other : int or float
           The value to add to the current progress level.
       Returns:
       -------
       ProgressBar
           The updated ProgressBar object.
       """
       assert type(other) in [float, int], "can only add a number"
       self.setAndPlot(self._level + other)
       return self
   def __sub__(self, other):
       """
       Decrement the progress bar by a specified amount.
       Parameters:
       ----------
       other : int or float
           The value to subtract from the current progress level.
       Returns:
       -------
       ProgressBar
           The updated ProgressBar object.
       """
       return self.__add__(-other)
   def __iadd__(self, other):
       """
       In-place increment of the progress bar by a specified amount.
       Parameters:
       ----------
       other : int or float
           The value to add to the current progress level.
       Returns:
       -------
       ProgressBar
           The updated ProgressBar object.
       """
       return self.__add__(other)
   def __isub__(self, other):
       """
       In-place decrement of the progress bar by a specified amount.
       Parameters:
       ----------
       other : int or float
           The value to subtract from the current progress level.
       Returns:
       -------
       ProgressBar
           The updated ProgressBar object.
       """
       return self.__add__(-other)
   def __del__(self):
       """
       Destructor method. Ensures a new line is printed after the progress bar is completed.
       """
       sys.stdout.write("\n")