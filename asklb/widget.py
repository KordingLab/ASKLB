"""
IPyWidget implementation of ASKLB.
"""
# built-in modules
import configparser
import copy
import datetime
from io import BytesIO
import os
import sys
import time
import threading
import warnings

# widget modules
import ipywidgets as widgets
from ipywidgets import Box

# ML modules
from autosklearn.classification import AutoSklearnClassifier
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import sklearn.model_selection

# Authentication and database modules
import gridfs
import pymongo
import bcrypt
import base64

# asklb modules
import model_utils

"""Model fitting functions"""

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
# Colab configuration
config.read(".widget_config.ini")
#config.read("config/widget_config.ini")
MONGO_PWD = base64.decodebytes(config['DEFAULT']['mongo_pwd'].encode("UTF-8")).decode().strip()
MONGO_URI = config['DEFAULT']['mongo_uri'].format(MONGO_PWD) #replace with remote db URI
MAX_TIME = int(config['DEFAULT']['max_time'])
MAX_BUDGET = int(config['DEFAULT']['max_budget'])
TRAIN_SIZE = float(config['DEFAULT']['train_size'])

class ASKLBWidget(Box):
    """
    ASKLB Widget, extends ipywidget's Box.
    """

    def __init__(self, textbox_upload=True, **kwargs):
        """Initializes ASKLBWidget by creating all widget components."""

        # TODO hack since FileUpload widgets do not work on any notebook hosting services (Azure, Colab)
        self.textbox_upload = textbox_upload

        # Set up authentication to the database
        client = pymongo.MongoClient(MONGO_URI)
        self.db = client.asklb_test
        self.username = None

        self.queries = 0
        # We make the assumption that the first column are the labels.
        # TODO can add "checksum" for the y's for the data to be in the same order
        self.data = []
        self.filenames = []
        self.models = []
        # We make the assumption that the data are uploaded in the same order.
        self.train_idxs = []
        self.test_idxs = []

        # Widgets for user authentication
        self.user_text_widget = widgets.Text(
            placeholder='Username',
            description='Username:'
        )

        self.password_text_widget = widgets.Password(
            placeholder='Password',
            description='Password:'
        )

        self.sign_in_widget = widgets.Button(
            description="Sign In",
            layout = widgets.Layout(width='auto'),
            button_style='primary',
            disabled=False)
        self.sign_in_widget.on_click(self.on_sign_in_widget_click)

        self.register_widget = widgets.Button(
            description="Register",
            layout = widgets.Layout(width='auto'),
            button_style='primary',
            disabled=False)
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

        if self.textbox_upload:
            self.upload_text = widgets.Text(
                placeholder="CSV filename here")
            self.test_size_text = widgets.Text(
                placeholder="Input test set size here")
            self.upload_button = widgets.Button(
                description="Upload Data",
            layout = widgets.Layout(width='auto'),
            button_style='primary',
            disabled=False)
            self.upload_button.on_click(self.on_upload_button_clicked)

            self.upload_widget = widgets.HBox([self.upload_text, self.test_size_text, self.upload_button])

        else:
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

        self.models_accordian = widgets.Accordion(children=[self.metrics_output_widget,
                                                       self.model_output_widget])
        self.models_accordian.set_title(0, 'Performance Metrics')
        self.models_accordian.set_title(1, 'Models and Weights Data')

        main_layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            border='solid',
            width='80%')

        auth_widget_items = [self.user_text_widget, self.password_text_widget, self.sign_in_widget, self.register_widget]
        self.auth_widget = widgets.VBox([widgets.HBox(auth_widget_items), self.auth_label_widget])

        self.tab_nest = widgets.Tab()
        self.tab_nest.children = [self.models_accordian]
        self.tab_nest.set_title(0, "Model Run Info")

        automl_widget_items = [runtime_slider,
                               budget_slider,
                               self.upload_widget,
                               self.fit_button_widget,
                               self.progress_widget,
                               self.event_output_widget,
                               self.tab_nest]

        self.automl_widget = widgets.VBox(automl_widget_items)
        self.automl_widget.layout.visibility = 'hidden'

        widget_items = [self.auth_widget, self.automl_widget]

        super(Box, self).__init__(children=widget_items, layout=main_layout, **kwargs)


    def on_upload_button_clicked(self, button):
        """Defines widget behavior after the upload button has been clicked.

        Side effects:
            - disables upload_button
            - disables upload_text
            - disables test_size_text on the first dataset uploaded
            - enables fit_button_widget
            - appends new data to data list
        """

        #data_array = np.loadtxt(open(self.upload_text.value, "rb"), delimiter=",")
        # we assume that the data are uploaded without a header
        data_array = pd.read_csv(self.upload_text.value, header=None)
        self.data.append(data_array)
        self.filenames.append(self.upload_text.value)

        # this is the first dataset loaded
        if len(self.data) == 1:
            n_samples = self.data[0].shape[0]
            indices = np.arange(n_samples)

            test_size = int(self.test_size_text.value)
            assert test_size < n_samples, "The specified test size is greater than the sample size!"
            split_idx = -1 * test_size

            self.train_idxs = indices[:split_idx]
            self.test_idxs = indices[split_idx:]
            self.test_size_text.disabled = True

        with self.event_output_widget:
            print("DATA PROCESSING COMPLETE.")

        self.upload_button.disabled = True
        self.upload_text.disabled = True
        self.fit_button_widget.disabled = False


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
                    self.username = username
                else:
                    self.auth_label_widget.value = "Incorrect password."
            else:
                self.auth_label_widget.value = "No user found. First time users must register."
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
                users.insert_one({'username' : username, 'password' : hashpass})
                self.auth_label_widget.value = "Registered successfully! You may now sign in."
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

        automl_args = {}

        automl_args['time_left_for_this_task'] = run_time
        # TODO functionality to load this from Mongo
        automl_args['metadata_directory'] = ".metalearning/metalearning_files/"
        #automl_args['metadata_directory'] = "../metalearning/metalearning_files/"

        automl = AutoSklearnClassifier(**automl_args)
        thread = threading.Thread(target=self.update_progress,
                                  args=(self.progress_widget,))
        thread.start()

        # always load a copy of the latest dataset
        cur_data = self.data[-1].copy()

        y = cur_data.pop(0)
        X, feat_types, _ = model_utils.process_feat_types(cur_data)

        X_train = X.iloc[self.train_idxs]
        y_train = y.iloc[self.train_idxs]

        X_test = X.iloc[self.test_idxs]
        y_test = y.iloc[self.test_idxs]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with HiddenPrints():
                automl.fit(X_train, y_train)#feat_type=feat_types)

        # Automl has finished fitting:
        self.models.append(copy.deepcopy(automl))

        with self.event_output_widget:
            print("FITTING COMPLETED WITH FITTING TIME PARAMETER AS ", int(run_time/60), " MINUTES")

        with self.metrics_output_widget:
            y_train_hat = automl.predict(X_train)
            train_accuracy_score = metrics.accuracy_score(y_train, y_train_hat)

            y_test_hat = automl.predict(X_test)
            test_accuracy_score = metrics.accuracy_score(y_test, y_test_hat)

            thresholdout_score = model_utils.thresholdout(train_accuracy_score, test_accuracy_score)

            output_str = "Run {}: train acc: {:.4}, noised test acc: {:.4}\n".format(self.queries, train_accuracy_score, thresholdout_score)
            print(output_str)

        with self.model_output_widget:
            print("MODELS:")
            print(automl.get_models_with_weights())

        if self.textbox_upload:
            self.upload_button.disabled = False
            self.upload_text.disabled = False
        else:
            self.upload_widget.disabled = False

        if self.queries == self.budget_widget.value:
            self.on_budget_completion()

        return automl


    def update_progress(self, progress):
        """Updates progress widget"""
        for i in range(int(progress.max/5)):
            time.sleep(5)
            progress.value = progress.value+5


    def on_budget_completion(self):
        """Defines widget behavior when the query budget is exhausted.

        Side Effects:
        - creates "Final Model Selection" tab
            - final model selection dropdown
            - final model selection button
            - final model results output
        - disables upload_widget
        - disables fit_button_widget

        """
        if self.textbox_upload:
            self.upload_button.disabled = False
            self.upload_text.disabled = False
        else:
            self.upload_widget.disabled = True

        self.fit_button_widget.disabled = True

        with self.event_output_widget:
            print("QUERY LIMIT MET.")
            print("SELECT FINAL MODEL.")

        self.final_model_dropdown = widgets.Dropdown(
            options = [("Model {}".format(i), i) for i in range(1, self.queries+1)],
            disabled=False
        )

        self.final_model_button = widgets.Button(
            description="Confirm Final Model Choice",
            layout = widgets.Layout(width='auto'),
            button_style='primary',
            disabled=False)
        self.final_model_button.on_click(self.on_final_model_button_clicked)

        self.final_output_widget = widgets.Output(layout={'border': '1px solid black'})

        final_model_tab = widgets.VBox([self.final_model_dropdown,
                                        self.final_model_button,
                                        self.final_output_widget])

        # TODO some way of appending to current tab children? Re-setting children here is ugly
        self.tab_nest.children = [self.models_accordian, final_model_tab]
        self.tab_nest.set_title(0, "Model Run Info")
        self.tab_nest.set_title(1, "Select Final Model")


    def on_final_model_button_clicked(self, button):
        """Displays final model information when a model is selected.

        Side Effects:
            - disables final_model_button
            - disables final_model_dropdown

        """
        self.final_model_button.disabled = True
        self.final_model_dropdown.disabled = True

        with self.final_output_widget:
            print("Chosen model true test performance:")
            # for off by 1 query indexing
            model_idx = self.final_model_dropdown.value - 1
            sel_model = self.models[model_idx]
            sel_filename = self.filenames[model_idx]
            sel_data = self.data[model_idx].copy()

            # TODO move to model_utils function
            y = sel_data.pop(0)
            X, _, _ = model_utils.process_feat_types(sel_data)

            X_test = X.iloc[self.test_idxs]
            y_test = y.iloc[self.test_idxs]

            y_test_hat = sel_model.predict(X_test)
            y_test_prob = sel_model.predict_proba(X_test)[:,1]
            test_accuracy_score = metrics.accuracy_score(y_test, y_test_hat)
            test_auc_score = metrics.roc_auc_score(y_test, y_test_prob)
            output_str = "Accuracy: {:.4}\nAUC: {:.4}".format(test_accuracy_score, test_auc_score)
            print(output_str)

            # write chosen dataset to mongo
            fs = gridfs.GridFS(self.db)
            dataset_id = fs.put(open(sel_filename, "rb"))
            #print(dataset_id)

            dataset_info = {
                "user": self.username,
                "date": datetime.datetime.utcnow(),
                "max_budget": self.queries,
                "train_idx":  self.train_idxs.tolist(),
                "test_idx": self.test_idxs.tolist(),
                "sample_size": X.shape[0],
                "test_auc": test_auc_score,
                "test_acc": test_accuracy_score,
                "filename": sel_filename,
                "gridfs_id": dataset_id
            }

            doc_id = self.db.datasets.insert_one(dataset_info).inserted_id
            #print(doc_id)
