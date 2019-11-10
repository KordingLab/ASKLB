"""
IPyWidget implementation of ASKLB.
"""
# built-in modules
import os
import sys
import time
import threading
import warnings

# widget modules
import ipywidgets as widgets
from ipywidgets import Box

# ML modules
import autosklearn.classification
import numpy as np
import sklearn.model_selection

def thresholdout(train_acc, test_acc, threshold=0.01, noise=0.03):
    """
    Applies the thresholdout algorithm to produce a (possibly) noised output.
    
    An implementation of the algorithm presented in Section 3 of "Holdout Reuse."
    
    Args:
        train_acc (float)
        test_acc (float)
        threshold (float): the base difference between train and test accuracies for thresholdout to apply
        noise (float): the noise rate for the Laplacian noise applied
    """
    threshold_hat = threshold + np.random.laplace(0, 2*noise)
    
    if np.abs(train_acc - test_acc) > (threshold_hat + np.random.laplace(0, 4*noise)):
        return test_acc + np.random.laplace(0, noise)
    else:
        return train_acc


class HiddenPrints:
    """
    Class for catching and hiding print statements
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


"""Constants"""
MAX_TIME = 60
MAX_BUDGET = 10

class ASKLBWidget(Box):
    """
    ASKLB Widget, extends ipywidget's Box.
    """
    queries = 0

    def __init__(self, **kwargs):
        """Initializes ASKLBWidget by creating all widget components."""

        self.runtime_widget = widgets.IntSlider(
            value=1,
            min=1,
            max=MAX_TIME)
        
        self.budget_widget = widgets.IntSlider(
            value=int(MAX_BUDGET/2),
            min=1,
            max=MAX_BUDGET)

        self.progress_widget = widgets.IntProgress(
            value=0, 
            min=0, 
            description="Progress")

        self.fit_button_widget = widgets.Button(
            description="Fit Data to AutoML Model", 
            layout = widgets.Layout( width='auto'),
            button_style='primary',
            disabled=True)

        self.event_output_widget = widgets.Output(layout={'border': '1px solid black'})
        self.model_output_widget = widgets.Output()
        self.metrics_output_widget = widgets.Output()

        self.assemble_widget(**kwargs)

    def assemble_widget(self, **kwargs):
        """Assembles the individual widget components in a single Box instance."""
        
        runtime_slider = widgets.HBox([widgets.Label('Run time (min):'), self.runtime_widget])
        budget_slider = widgets.HBox([widgets.Label('Query budget:'), self.budget_widget])

        models_accordian = widgets.Accordion(children=[self.metrics_output_widget, 
                                                       self.model_output_widget])
        models_accordian.set_title(0, 'Performance Metrics')
        models_accordian.set_title(1, 'Models and Weights Data')

        self.layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            border='solid',
            width='65%')

        widget_items = [runtime_slider, 
                        budget_slider, 
                        #self.upload_widget, 
                        self.fit_button_widget, 
                        self.progress_widget, 
                        self.event_output_widget, 
                        models_accordian]

        super(Box, self).__init__(children=widget_items, layout=self.layout, **kwargs)

    def on_fit_button_clicked(self, *args):
        """Widget behavior after fit button click event occurs."""
        

def work(progress):
        for i in range(int(progress.max/5)):
            time.sleep(5)
            progress.value = progress.value+5