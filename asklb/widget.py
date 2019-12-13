"""
IPyWidget implementation of ASKLB.
"""
# built-in modules
from io import BytesIO
import os
import sys
import time
import threading
import warnings
import configparser

# widget modules
import ipywidgets as widgets
from ipywidgets import Box

# ML modules
import autosklearn.classification
import numpy as np
import sklearn.metrics as metrics
import sklearn.model_selection

# Authentication and database modules
import pymongo
import bcrypt

"""Model fitting functions"""

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
config = configparser.ConfigParser()
config.read("../config/widget_config.ini")
MONGO_URI = config['DEFAULT']['mongo_uri'] #replace with remote db URI
MAX_TIME = int(config['DEFAULT']['max_time'])
MAX_BUDGET = int(config['DEFAULT']['max_budget'])
TRAIN_SIZE = float(config['DEFAULT']['train_size'])

class ASKLBWidget(Box):
    """
    ASKLB Widget, extends ipywidget's Box.
    """

    def __init__(self, **kwargs):
        """Initializes ASKLBWidget by creating all widget components."""

        # Set up authentication to the database
        client = pymongo.MongoClient(MONGO_URI)
        self.db = client.asklb_test

        self.queries = 0
        # We make the assumption that the first column are the labels.
        # TODO can add "checksum" for the y's for the data to be in the same order
        self.data = []
        # We make the assumption that the data are uploaded in the same order.
        self.train_idxs = []
        self.test_idxs = []

        # Widgets for user authentication
        self.user_text_widget = widgets.Text(
            placeholder='Username',
            description='Username:'
        )

        self.password_text_widget = widgets.Text(
            placeholder='Password',
            description='Password:'
        )

        self.sign_in_widget = widgets.Button(
            description="Sign In", 
            layout = widgets.Layout(width='auto'),
            button_style='primary',
            disabled=False) # init with fit button disabled
        self.sign_in_widget.on_click(self.on_sign_in_widget_click)

        self.register_widget = widgets.Button(
            description="Register", 
            layout = widgets.Layout(width='auto'),
            button_style='primary',
            disabled=False) # init with fit button disabled
        self.register_widget.on_click(self.on_register_widget_click)

        self.auth_label_widget = widgets.Label(value="Please sign in or register")

        self.is_signed_in = False #Boolean for being signed in

        # Widgets for automl runtime
        self.runtime_widget = widgets.IntSlider(
            value=1,
            min=1,
            max=MAX_TIME)
        
        self.budget_widget = widgets.IntSlider(
            value=int(MAX_BUDGET/2),
            min=1,
            max=MAX_BUDGET)

        self.upload_widget = widgets.FileUpload(
            accept='.csv',  # Accepted file extension
            multiple=False  # Only accept a single file
        )
        self.upload_widget.observe(self.on_data_upload_begin, names="_counter")
        self.upload_widget.observe(self.on_data_upload_completion, names="value")

        self.progress_widget = widgets.IntProgress(
            value=0, 
            min=0, 
            description="Progress")

        self.fit_button_widget = widgets.Button(
            description="Fit Data to AutoML Model", 
            layout = widgets.Layout(width='auto'),
            button_style='primary',
            disabled=True) # init with fit button disabled
        self.fit_button_widget.on_click(self.on_fit_button_clicked)

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

        main_layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            border='solid',
            width='80%')

        auth_widget_items = [self.user_text_widget, self.password_text_widget, self.sign_in_widget, self.register_widget]
        self.auth_widget = widgets.VBox([widgets.HBox(auth_widget_items), self.auth_label_widget])

        automl_widget_items = [runtime_slider, 
                               budget_slider, 
                               self.upload_widget, 
                               self.fit_button_widget, 
                               self.progress_widget, 
                               self.event_output_widget, 
                               models_accordian]

        self.automl_widget = widgets.VBox(automl_widget_items)
        self.automl_widget.layout.visibility = 'hidden'

        widget_items = [self.auth_widget, self.automl_widget]

        super(Box, self).__init__(children=widget_items, layout=main_layout, **kwargs)


    def on_data_upload_completion(self, change_dict):
        """Defines widget behavior after a file has been uploaded.

        Processes the uploaded data.

        Side effects:
            - disables upload_widget
            - enables fit_button_widget
            - appends new data to data list

        Args:
            change_dict (dict): the value dict passed from the FileUpload widget.
        """
        #with self.event_output_widget:
        #    print(change_dict)
        with self.event_output_widget:
            print("DATA UPLOAD COMPLETE.")

        self.upload_widget.disabled = True
        self.fit_button_widget.disabled = False
        # https://github.com/jupyter-widgets/ipywidgets/issues/2538
        uploaded_filename = next(iter(self.upload_widget.value))
        b_stream = BytesIO(self.upload_widget.value[uploaded_filename]['content'])
        data_array = np.loadtxt(b_stream, delimiter=',')

        self.data.append(data_array)

        # this is the first dataset loaded
        if len(self.data) == 1:
            n_samples = self.data[0].shape[0]
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            split_idx = int(n_samples * TRAIN_SIZE)
            self.train_idxs = indices[:split_idx]
            self.test_idxs = indices[split_idx:]            

    def on_data_upload_begin(self, counter):
        """Defines widget behavior once a file has begun uploading."""
        #TODO figure out FileUpload observation *before* upload completion
        pass
        #with self.event_output_widget:
        #    print("FILE UPLOAD BEGUN.")

    def on_sign_in_widget_click(self, button):
        """Defines widget behavior after the user clicks on the sign-in button

        Side effects:
            - enables the AutoML widget
            - TODO switches the sign-in widget with a signed-in widget

        Args:
            button (widgets.Button): the sign-in object clicked.
        """        

        if not self.is_signed_in:
            users = self.db.users
            username = self.user_text_widget.value
            password = self.password_text_widget.value

            login_user = users.find_one({'username' : username})

            # Upon sign in, disable auth widgets and show automl widgets
            if login_user:
                if bcrypt.hashpw(password.encode('utf-8'), 
                                login_user['password']) == login_user['password']:
                    self.auth_label_widget.value = "Authentication successful!"
                    self.is_signed_in = True
                    self.automl_widget.layout.visibility = 'visible'
                    self.sign_in_widget.description = "Sign out"
                    self.user_text_widget.disabled = True
                    self.password_text_widget.disabled = True
                    self.register_widget.disabled = True
                else:
                    self.auth_label_widget.value = "Incorrect password!"   
            else:
                self.auth_label_widget.value = "No user found!"   
        else:
            # Upon sign out, enable auth widgets and hide automl widgets
            self.auth_label_widget.value = "Signed out!"
            self.is_signed_in = False
            self.automl_widget.layout.visibility = 'hidden'
            self.sign_in_widget.description = "Sign in"
            self.user_text_widget.disabled = False
            self.password_text_widget.disabled = False
            self.register_widget.disabled = False


    def on_register_widget_click(self, button):
        """Defines widget behavior after the user clicks on the sign-in button

        Side effects:
            - enables the AutoML widget
            - TODO switches the sign-in widget with a signed-in widget

        Args:
            button (widgets.Button): the sign-in object clicked.
        """

        users = self.db.users
        username = self.user_text_widget.value
        password = self.password_text_widget.value

        if not username:
            self.auth_label_widget.value = "Username is required."
        elif not password:
            self.auth_label_widget.value = "Password is required."
        else:            
            existing_user = users.find_one({'username' : username})        

            if existing_user is None:
                hashpass = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                users.insert({'username' : username, 'password' : hashpass})
                self.auth_label_widget.value = "Registered successfully!"
            else:
                self.auth_label_widget.value = "That username exists!"


    def on_fit_button_clicked(self, button):
        """Widget behavior after fit button click event occurs.
        
        Side effects:
            - disables budget_widget
            - disables fit_button_widget

        Args: 
            button (widgets.Button): the button object clicked.
        """
        
        self.budget_widget.disabled = True
        self.fit_button_widget.disabled = True
        # TODO figure out model display management
        self.model_output_widget.clear_output()

        final_runtime_value_seconds = self.runtime_widget.value * 60
        self.progress_widget.value = 0    
        self.progress_widget.max = final_runtime_value_seconds

        if self.queries == self.budget_widget.value:
            with self.event_output_widget:
                print("QUERY LIMIT MET.")
        else:
            self.queries += 1

            with self.event_output_widget:
                print("AUTOML RUN {} STARTED, FITTING TIME IS ".format(self.queries), 
                      int(final_runtime_value_seconds/60), " MINUTES")

            self.fit_automl(final_runtime_value_seconds)


    def fit_automl(self, run_time):
        """Runs auto-sklearn on the uploaded data and prints results.
        
        Side effects:
            - Enables upload_widget

        Args:
            run_time (int): The run time for auto-sklearn in seconds.
        Returns:
            automl (AutoSklearnClassifier): fitted auto-sklearn model.
        """

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task = run_time)
        thread = threading.Thread(target=self.update_progress, 
                                  args=(self.progress_widget,))    
        thread.start()

        # always load the latest dataset
        cur_data = self.data[-1]
        
        y = cur_data[:, 0]
        X = cur_data[:, 1:]

        X_train = X[self.train_idxs]
        y_train = y[self.train_idxs]

        X_test = X[self.test_idxs]
        y_test = y[self.test_idxs]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with HiddenPrints():
                automl.fit(X_train, y_train)

        with self.event_output_widget:
            print("FITTING COMPLETED WITH FITTING TIME PARAMETER AS ", int(run_time/60), " MINUTES")

        with self.metrics_output_widget:           
            y_train_hat = automl.predict(X_train)
            train_accuracy_score = metrics.accuracy_score(y_train, y_train_hat)
            
            y_test_hat = automl.predict(X_test)
            test_accuracy_score = metrics.accuracy_score(y_test, y_test_hat)

            thresholdout_score = thresholdout(train_accuracy_score, test_accuracy_score)

            output_str = "train acc: {}, test acc: {}\n".format(train_accuracy_score, thresholdout_score)
            print(output_str)

        with self.model_output_widget:
            print("MODELS:")
            print(automl.get_models_with_weights())

        self.upload_widget.disabled = False
        #self.fit_button_widget.disabled = False

        return automl
    
    def update_progress(self, progress):
        """Updates progress widget"""
        for i in range(int(progress.max/5)):
            time.sleep(5)
            progress.value = progress.value+5