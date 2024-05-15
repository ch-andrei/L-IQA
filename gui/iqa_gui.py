from utils.image_processing.image_tools import *
from utils.misc.miscelaneous import float2str3
from DisplayModels.display_model import DisplayModel
from DisplayModels.display_model_simul import DisplayDegradationModel, new_simul_params
from iqa_metrics.iqa_tool import IqaTool
from iqa_metrics.metrics import (iqa_msssim, iqa_lpips, iqa_vtamiq)
from iqa_metrics.vtamiq.vtamiq_wrapper import compute_pairwise_pref

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QTextCursor, QFont, QPalette, QColor
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QSpinBox, QDoubleSpinBox, QTextEdit, QVBoxLayout, QWidget)
from gui.qimageviewer import QImageViewer
from qimage2ndarray import rgb_view
import sys


class Window(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self, window_name, viewer):
        super().__init__()
        layout = QVBoxLayout()
        self.label = QLabel(window_name)
        layout.addWidget(self.label)
        layout.addWidget(viewer)
        self.setLayout(layout)
        # self.setGeometry(500,500,1000,800)


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class App(QWidget):
    def __init__(self):
        super().__init__()

        msssim_metric_name = "Contrast-based metric"
        lpips_metric_name = "AI-based metric"
        vtamiq_metric_name = "AI-based preference metric"

        self.metric_name_dict = {
            iqa_lpips.name: lpips_metric_name,
            iqa_msssim.name: msssim_metric_name,
            iqa_vtamiq.name: vtamiq_metric_name,
        }

        # Common parameters
        display_gamma_label = "Gamma"
        display_contrast_label = "Contrast ratio"
        display_l_min_label = "Lmin"
        display_l_unit_label = "cd/m2"
        display_l_max_label = "Lmax"
        age_simulation_label = "Observer's age"
        illumination_unit_label = "lux"
        illumination_label = "Illumination"
        screen_brightness = "Screen brightness"
        use_same_as_reference = "Use same as Reference"
        apply_reflection = "Apply reflection"
        apply_screen_dimming = "Apply screen dimming"
        apply_age_simulation = "Apply age simulation"
        load_image = 'Load Image'
        self.iqa_list = []

        # Reference parameters
        self.reference_group_box_layout = QVBoxLayout()
        self.reference_group_box = QGroupBox("Reference Parameters")
        self.reference_evaluation_options_group_box_layout = QVBoxLayout()
        self.age_simulation_options_group_box_layout = QVBoxLayout()
        self.apply_screen_dimming_reference = QCheckBox(apply_screen_dimming)
        self.apply_reflection_reference = QCheckBox(apply_reflection)
        self.apply_age_simulation = QCheckBox(apply_age_simulation)
        self.reference_evaluation_options_group_box = QGroupBox("Reference Evaluation Options")
        self.age_simulation_options_group_box = QGroupBox("Age Simulation Options")
        self.reference_image_group_box_layout = QVBoxLayout()
        self.button_load_reference_image = QPushButton(load_image)
        self.reference_image_view = QImageViewer()
        self.reference_image_group_box = QGroupBox("Reference Image")
        self.window_reference_image_view = None
        self.image_reference = None
        self.reference_display_group_box_layout = QGridLayout()
        self.display_reflectivity_label = "Reflectivity"
        self.reference_display_reflectivity_label = QLabel(self.display_reflectivity_label)
        self.reference_display_group_box = QGroupBox("Reference Display")
        self.reference_display_reflectivity = QDoubleSpinBox(self.reference_display_group_box)
        self.reference_display_gamma_label = QLabel(display_gamma_label)
        self.reference_display_gamma = QDoubleSpinBox(self.reference_display_group_box)
        self.reference_display_contrast_label = QLabel(display_contrast_label)
        self.reference_display_contrast = QSpinBox(self.reference_display_group_box)
        self.reference_display_l_min_label = QLabel(display_l_min_label)
        self.reference_display_l_min = QSpinBox(self.reference_display_group_box)
        self.reference_display_l_max_unit_label = QLabel(display_l_unit_label)
        self.reference_display_l_min_unit_label = QLabel(display_l_unit_label)
        self.reference_display_l_max_label = QLabel(display_l_max_label)
        self.reference_display_l_max = QSpinBox(self.reference_display_group_box)
        self.age_simulation_age_value_label = QLabel(age_simulation_label)
        self.age_simulation_age_value = QSpinBox(self.reference_display_group_box)
        self.reference_illumination_condition_group_box_layout = QGridLayout()
        self.reference_illumination_unit_label = QLabel(illumination_unit_label)
        self.reference_illumination_label = QLabel(illumination_label)
        self.reference_illumination_condition_group_box = QGroupBox("Reference Illumination Condition")
        self.reference_illumination_condition = QSpinBox(self.reference_illumination_condition_group_box)
        self.reference_target_screen_brightness_unit_label = QLabel(display_l_unit_label)
        self.reference_target_screen_brightness_label = QLabel(screen_brightness)
        self.reference_target_screen_brightness = QLabel('')
        self.dm_ref = None
        self.ddm_ref = None
        self.window_simulated_reference_image_view = None
        self.simulated_reference_image_number = 0
        self.simulated_reference_image_view = QImageViewer()

        # Test 1 parameters
        self.test_1_group_box_layout = QVBoxLayout()
        self.test_1_group_box = QGroupBox("Test 1 Parameters")
        self.test_1_evaluation_options_group_box_layout = QVBoxLayout()
        self.apply_screen_dimming_test_1 = QCheckBox(apply_screen_dimming)
        self.apply_reflection_test_1 = QCheckBox("Apply reflection")
        self.test_1_evaluation_options_same_as_reference = QCheckBox(use_same_as_reference)
        self.test_1_evaluation_options_group_box = QGroupBox("Test 1 Evaluation Options")
        self.test_1_image_group_box_layout = QVBoxLayout()
        self.button_load_test_1_image = QPushButton(load_image)
        self.test_1_image_view = QImageViewer()
        self.test_1_image_group_box = QGroupBox("Test 1 Image")
        self.window_test_1_image_view = None
        self.test_1_display_group_box_layout = QGridLayout()
        self.test_1_display_group_box = QGroupBox("Test 1 Display")
        self.test_1_display_reflectivity = QDoubleSpinBox(self.test_1_display_group_box)
        self.test_1_display_gamma = QDoubleSpinBox(self.test_1_display_group_box)
        self.test_1_display_contrast = QSpinBox(self.test_1_display_group_box)
        self.test_1_display_l_min = QSpinBox(self.test_1_display_group_box)
        self.test_1_display_l_max = QSpinBox(self.test_1_display_group_box)
        self.test_1_display_same_as_reference = QCheckBox(use_same_as_reference)
        self.test_1_illumination_condition_group_box_layout = QGridLayout()
        self.test_1_illumination_same_as_reference = QCheckBox(use_same_as_reference)
        self.test_1_illumination_condition_group_box = QGroupBox("Test 1 Illumination Condition")
        self.test_1_illumination_condition = QSpinBox(self.test_1_illumination_condition_group_box)
        self.test_1_display_reflectivity_label = QLabel(self.display_reflectivity_label)
        self.test_1_display_gamma_label = QLabel(display_gamma_label)
        self.test_1_display_contrast_label = QLabel(display_contrast_label)
        self.test_1_display_l_min_label = QLabel(display_l_min_label)
        self.test_1_display_l_max_unit_label = QLabel(display_l_unit_label)
        self.test_1_display_l_min_unit_label = QLabel(display_l_unit_label)
        self.test_1_display_l_max_label = QLabel(display_l_max_label)
        self.test_1_target_screen_brightness_unit_label = QLabel(display_l_unit_label)
        self.test_1_target_screen_brightness_label = QLabel(screen_brightness)
        self.test_1_target_screen_brightness = QLabel('')
        self.image_test_1 = None
        self.test_1_illumination_unit_label = QLabel(illumination_unit_label)
        self.test_1_illumination_label = QLabel(illumination_label)
        self.ddm_test_1 = None
        self.dm_test_1 = None
        self.window_simulated_test_1_image_view = None
        self.simulated_test_1_image_number = 0
        self.simulated_test_1_image_view = QImageViewer()

        # Test 2 parameters
        self.test_2_group_box_layout = QVBoxLayout()
        self.test_2_group_box = QGroupBox("Test 2 Parameters")
        self.test_2_evaluation_options_group_box_layout = QVBoxLayout()
        self.apply_screen_dimming_test_2 = QCheckBox(apply_screen_dimming)
        self.apply_reflection_test_2 = QCheckBox(apply_reflection)
        self.test_2_evaluation_options_same_as_reference = QCheckBox(use_same_as_reference)
        self.test_2_evaluation_options_group_box = QGroupBox("Test 2 Evaluation Options")
        self.test_2_image_group_box_layout = QVBoxLayout()
        self.button_load_test_2_image = QPushButton(load_image)
        self.test_2_image_view = QImageViewer()
        self.test_2_image_group_box = QGroupBox("Test 2 Image")
        self.window_test_2_image_view = None
        self.test_2_display_group_box_layout = QGridLayout()
        self.test_2_display_group_box = QGroupBox("Test 2 Display")
        self.test_2_display_reflectivity = QDoubleSpinBox(self.test_2_display_group_box)
        self.test_2_display_gamma = QDoubleSpinBox(self.test_2_display_group_box)
        self.test_2_display_contrast = QSpinBox(self.test_2_display_group_box)
        self.test_2_display_l_min = QSpinBox(self.test_2_display_group_box)
        self.test_2_display_l_max = QSpinBox(self.test_2_display_group_box)
        self.test_2_display_same_as_reference = QCheckBox(use_same_as_reference)
        self.test_2_illumination_condition_group_box_layout = QGridLayout()
        self.test_2_illumination_same_as_reference = QCheckBox(use_same_as_reference)
        self.test_2_illumination_condition_group_box = QGroupBox("Test 2 Illumination Condition")
        self.test_2_illumination_condition = QSpinBox(self.test_2_illumination_condition_group_box)
        self.test_2_display_reflectivity_label = QLabel(self.display_reflectivity_label)
        self.test_2_display_gamma_label = QLabel(display_gamma_label)
        self.test_2_display_contrast_label = QLabel(display_contrast_label)
        self.test_2_display_l_min_label = QLabel(display_l_min_label)
        self.test_2_display_l_max_unit_label = QLabel(display_l_unit_label)
        self.test_2_display_l_min_unit_label = QLabel(display_l_unit_label)
        self.test_2_display_l_max_label = QLabel(display_l_max_label)
        self.test_2_target_screen_brightness_unit_label = QLabel(display_l_unit_label)
        self.test_2_target_screen_brightness_label = QLabel(screen_brightness)
        self.test_2_target_screen_brightness = QLabel('')
        self.image_test_2 = None
        self.test_2_illumination_unit_label = QLabel(illumination_unit_label)
        self.test_2_illumination_label = QLabel(illumination_label)
        self.ddm_test_2 = None
        self.dm_test_2 = None
        self.window_simulated_test_2_image_view = None
        self.simulated_test_2_image_number = 0
        self.simulated_test_2_image_view = QImageViewer()

        # Metrics
        self.evaluation_parameters_group_box_layout = QGridLayout()
        self.evaluation_parameters_group_box = QGroupBox("Evaluation Parameters")
        self.evaluation_metrics_group_box_layout = QHBoxLayout()

        self.iqa_lpips = QCheckBox(lpips_metric_name)
        self.iqa_msssim = QCheckBox(msssim_metric_name)
        self.iqa_vtamiq = QCheckBox(vtamiq_metric_name)
        self.evaluation_metrics_group_box = QGroupBox("Evaluation metrics")

        # Results
        self.simulation_results_group_box_layout = QVBoxLayout()
        self.simulated_images_group_box_layout = QHBoxLayout()
        self.simulation_results_group_box = QGroupBox("Simulation Results")
        self.results_metrics_group_box_layout = QVBoxLayout()
        self.output_metric_computation = QTextEdit()
        self.button_compute_iqa_metrics = QPushButton('Compute IQA metric(s)')
        self.results_metrics_group_box = QGroupBox("Metrics results")
        self.simulated_test_1_group_box_layout = QVBoxLayout()
        self.button_simulate_test_1_image = QPushButton('Simulate Test 1 Image')
        self.simulated_test_1_group_box = QGroupBox("Simulated Test 1 Image")
        self.simulated_test_2_group_box_layout = QVBoxLayout()
        self.button_simulate_test_2_image = QPushButton('Simulate Test 2 Image')
        self.simulated_test_2_group_box = QGroupBox("Simulated Test 2 Image")
        self.simulated_reference_group_box_layout = QVBoxLayout()
        self.button_simulate_reference_image = QPushButton('Simulate Reference Image')
        self.simulated_reference_group_box = QGroupBox("Simulated Reference Image")
        self.simulated_images_group_box = QGroupBox("Simulated Images")

        ### Default parameters
        self.illumination_condition_max = 30000
        self.illumination_condition_min = 0
        self.illumination_condition_default_value = 1000

        self.display_l_max_max = 50000
        self.display_l_max_min = 0
        self.display_l_max_default_value = 459.2

        self.age_simulation_min_value = 24
        self.age_simulation_max_value = 99
        self.age_simulation_age_default_value = 24

        self.display_l_min_max = 50000
        self.display_l_min_min = 0
        self.display_l_min_default_value = 1.811

        self.display_contrast_max = 100000
        self.display_contrast_min = 1
        self.display_contrast_default_value = 1000

        self.display_gamma_max = 3
        self.display_gamma_min = 1.5
        self.display_gamma_default_value = 2.2

        self.display_reflectivity_max = 1
        self.display_reflectivity_min = 0
        self.display_reflectivity_default_value = 0.01

        # Initialization
        self.title = 'PIQA GUI'
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)

        main_layout = QGridLayout()

        self.create_evaluation_parameters_group_box()
        self.create_simulation_results_group_box()

        main_layout.addWidget(self.evaluation_parameters_group_box, 0, 0)
        main_layout.addWidget(self.simulation_results_group_box, 1, 0)

        sys.stdout = Stream(newText=self.on_update_text)

        self.setLayout(main_layout)
        self.show()

    def on_update_text(self, text):
        cursor = self.output_metric_computation.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.output_metric_computation.setTextCursor(cursor)
        self.output_metric_computation.ensureCursorVisible()

    def create_reference_illumination_condition_group_box(self):
        self.reference_illumination_condition.setRange(self.illumination_condition_min, self.illumination_condition_max)
        self.reference_illumination_condition.setValue(self.illumination_condition_default_value)
        self.reference_illumination_condition.valueChanged.connect(self.update_all_illumination_simulation_parameters)

        self.reference_illumination_label.setBuddy(self.reference_illumination_condition)
        self.reference_illumination_unit_label.setBuddy(self.reference_illumination_condition)

        self.reference_illumination_condition_group_box_layout.addWidget(self.reference_illumination_label, 0, 0)
        self.reference_illumination_condition_group_box_layout.addWidget(self.reference_illumination_condition, 0, 1)
        self.reference_illumination_condition_group_box_layout.addWidget(self.reference_illumination_unit_label, 0, 2)
        self.reference_illumination_condition_group_box_layout.addWidget(self.reference_target_screen_brightness_label,
                                                                         1, 0)
        self.reference_illumination_condition_group_box_layout.addWidget(self.reference_target_screen_brightness, 1, 1)
        self.reference_illumination_condition_group_box_layout.addWidget(
            self.reference_target_screen_brightness_unit_label, 1, 2)

        self.reference_illumination_condition_group_box.setLayout(
            self.reference_illumination_condition_group_box_layout)

    def create_reference_display_group_box(self):
        self.reference_display_l_max.setRange(self.display_l_max_min, self.display_l_max_max)
        self.reference_display_l_max.setValue(self.display_l_max_default_value)
        self.reference_display_l_max.valueChanged.connect(self.update_all_display)
        self.reference_display_l_max_label.setBuddy(self.reference_display_l_max)
        self.reference_display_l_max_unit_label.setBuddy(self.reference_display_l_max)

        self.reference_display_l_min.setRange(self.display_l_min_min, self.display_l_min_max)
        self.reference_display_l_min.setValue(self.display_l_min_default_value)
        self.reference_display_l_min.valueChanged.connect(self.update_all_display)
        self.reference_display_l_min_label.setBuddy(self.reference_display_l_min)
        self.reference_display_l_min_unit_label.setBuddy(self.reference_display_l_min)

        self.reference_display_contrast.setRange(self.display_contrast_min, self.display_contrast_max)
        self.reference_display_contrast.setValue(self.display_contrast_default_value)
        self.reference_display_contrast.valueChanged.connect(self.update_all_display)
        self.reference_display_contrast_label.setBuddy(self.reference_display_l_max)

        self.reference_display_gamma.setRange(self.display_gamma_min, self.display_gamma_max)
        self.reference_display_gamma.setValue(self.display_gamma_default_value)
        self.reference_display_gamma.valueChanged.connect(self.update_all_display)
        self.reference_display_gamma_label.setBuddy(self.reference_display_l_max)

        self.reference_display_reflectivity.setRange(self.display_reflectivity_min, self.display_reflectivity_max)
        self.reference_display_reflectivity.setValue(self.display_reflectivity_default_value)
        self.reference_display_reflectivity.valueChanged.connect(self.update_all_display)
        self.reference_display_reflectivity_label.setBuddy(self.reference_display_l_max)

        self.reference_display_group_box_layout.addWidget(self.reference_display_l_max_label, 0, 0)
        self.reference_display_group_box_layout.addWidget(self.reference_display_l_max, 0, 1)
        self.reference_display_group_box_layout.addWidget(self.reference_display_l_max_unit_label, 0, 2)
        self.reference_display_group_box_layout.addWidget(self.reference_display_l_min_label, 1, 0)
        self.reference_display_group_box_layout.addWidget(self.reference_display_l_min, 1, 1)
        self.reference_display_group_box_layout.addWidget(self.reference_display_l_min_unit_label, 1, 2)
        self.reference_display_group_box_layout.addWidget(self.reference_display_contrast_label, 2, 0)
        self.reference_display_group_box_layout.addWidget(self.reference_display_contrast, 2, 1)
        self.reference_display_group_box_layout.addWidget(self.reference_display_gamma_label, 3, 0)
        self.reference_display_group_box_layout.addWidget(self.reference_display_gamma, 3, 1)
        self.reference_display_group_box_layout.addWidget(self.reference_display_reflectivity_label, 4, 0)
        self.reference_display_group_box_layout.addWidget(self.reference_display_reflectivity, 4, 1)

        self.reference_display_group_box.setLayout(self.reference_display_group_box_layout)

    def load_reference_image(self):
        self.reference_image_view.open()
        self.image_reference = self.reference_image_view.getDisplayedImage()
        self.window_reference_image_view = Window("Reference image", self.reference_image_view)
        self.window_reference_image_view.show()

    def create_reference_image_group_box(self):
        self.button_load_reference_image.clicked.connect(self.load_reference_image)

        self.reference_image_group_box_layout.addWidget(self.button_load_reference_image)

        self.reference_image_group_box.setLayout(self.reference_image_group_box_layout)

    def create_reference_evaluation_options_group_box(self):
        self.apply_reflection_reference.setChecked(False)
        self.apply_reflection_reference.stateChanged.connect(self.update_all_illumination_simulation_parameters)
        self.apply_screen_dimming_reference.setChecked(True)
        self.apply_screen_dimming_reference.stateChanged.connect(self.update_all_illumination_simulation_parameters)

        self.reference_evaluation_options_group_box_layout.addWidget(self.apply_reflection_reference)
        self.reference_evaluation_options_group_box_layout.addWidget(self.apply_screen_dimming_reference)
        self.reference_evaluation_options_group_box_layout.addStretch(1)

        self.reference_evaluation_options_group_box.setLayout(self.reference_evaluation_options_group_box_layout)

    def create_age_simulation_options_group_box(self):
        self.apply_age_simulation.setChecked(False)
        self.apply_age_simulation.stateChanged.connect(self.update_age_simulation_parameters)

        self.age_simulation_options_group_box_layout.addWidget(self.apply_age_simulation)
        self.age_simulation_options_group_box_layout.addWidget(self.age_simulation_age_value)
        self.age_simulation_options_group_box_layout.addStretch(1)

        self.age_simulation_options_group_box.setLayout(self.age_simulation_options_group_box_layout)

        self.age_simulation_age_value.setRange(self.age_simulation_min_value, self.age_simulation_max_value)
        self.age_simulation_age_value.setValue(self.age_simulation_age_default_value)
        self.age_simulation_age_value.valueChanged.connect(self.update_age_simulation_parameters)
        self.age_simulation_age_value_label.setBuddy(self.age_simulation_age_value)

    def create_reference_group_box(self):
        self.create_reference_illumination_condition_group_box()
        self.create_reference_display_group_box()
        self.create_reference_image_group_box()
        self.create_reference_evaluation_options_group_box()
        self.create_age_simulation_options_group_box()

        self.reference_group_box_layout.addWidget(self.reference_illumination_condition_group_box)
        self.reference_group_box_layout.addWidget(self.reference_display_group_box)
        self.reference_group_box_layout.addWidget(self.reference_evaluation_options_group_box)
        self.reference_group_box_layout.addWidget(self.reference_image_group_box)
        self.reference_group_box_layout.addWidget(self.age_simulation_options_group_box)
        self.reference_group_box_layout.addStretch(1)

        self.reference_group_box.setLayout(self.reference_group_box_layout)

        self.update_reference_simulation_parameters()

    def create_test_1_illumination_condition_group_box(self):
        self.test_1_illumination_condition.setRange(self.illumination_condition_min, self.illumination_condition_max)
        self.test_1_illumination_condition.setValue(self.illumination_condition_default_value)
        self.test_1_illumination_condition.valueChanged.connect(self.update_test_1_illumination_simulation_parameters)
        self.test_1_illumination_label.setBuddy(self.test_1_illumination_condition)
        self.test_1_illumination_unit_label.setBuddy(self.test_1_illumination_condition)
        self.test_1_illumination_same_as_reference.setChecked(True)
        self.test_1_illumination_same_as_reference.stateChanged.connect(
            self.update_all_illumination_simulation_parameters)

        self.test_1_illumination_condition_group_box_layout.addWidget(self.test_1_illumination_same_as_reference, 0, 0,
                                                                      1, 3)
        self.test_1_illumination_condition_group_box_layout.addWidget(self.test_1_illumination_label, 1, 0)
        self.test_1_illumination_condition_group_box_layout.addWidget(self.test_1_illumination_condition, 1, 1)
        self.test_1_illumination_condition_group_box_layout.addWidget(self.test_1_illumination_unit_label, 1, 2)
        self.test_1_illumination_condition_group_box_layout.addWidget(self.test_1_target_screen_brightness_label, 2, 0)
        self.test_1_illumination_condition_group_box_layout.addWidget(self.test_1_target_screen_brightness, 2, 1)
        self.test_1_illumination_condition_group_box_layout.addWidget(self.test_1_target_screen_brightness_unit_label,
                                                                      2, 2)

        self.test_1_illumination_condition_group_box.setLayout(self.test_1_illumination_condition_group_box_layout)

    def create_test_2_illumination_condition_group_box(self):
        self.test_2_illumination_condition.setRange(self.illumination_condition_min, self.illumination_condition_max)
        self.test_2_illumination_condition.setValue(self.illumination_condition_default_value)
        self.test_2_illumination_condition.valueChanged.connect(self.update_test_2_illumination_simulation_parameters)
        self.test_2_illumination_label.setBuddy(self.test_2_illumination_condition)
        self.test_2_illumination_unit_label.setBuddy(self.test_2_illumination_condition)
        self.test_2_illumination_same_as_reference.setChecked(True)
        self.test_2_illumination_same_as_reference.stateChanged.connect(
            self.update_all_illumination_simulation_parameters)

        self.test_2_illumination_condition_group_box_layout.addWidget(self.test_2_illumination_same_as_reference, 0, 0,
                                                                      1, 3)
        self.test_2_illumination_condition_group_box_layout.addWidget(self.test_2_illumination_label, 1, 0)
        self.test_2_illumination_condition_group_box_layout.addWidget(self.test_2_illumination_condition, 1, 1)
        self.test_2_illumination_condition_group_box_layout.addWidget(self.test_2_illumination_unit_label, 1, 2)
        self.test_2_illumination_condition_group_box_layout.addWidget(self.test_2_target_screen_brightness_label, 2, 0)
        self.test_2_illumination_condition_group_box_layout.addWidget(self.test_2_target_screen_brightness, 2, 1)
        self.test_2_illumination_condition_group_box_layout.addWidget(self.test_2_target_screen_brightness_unit_label,
                                                                      2, 2)

        self.test_2_illumination_condition_group_box.setLayout(self.test_2_illumination_condition_group_box_layout)

    def create_test_1_display_group_box(self):
        self.test_1_display_same_as_reference.setChecked(True)
        self.test_1_display_same_as_reference.stateChanged.connect(self.update_all_display)

        self.test_1_display_l_max.setRange(self.display_l_max_min, self.display_l_max_max)
        self.test_1_display_l_max.setValue(self.display_l_max_default_value)
        self.test_1_display_l_max.valueChanged.connect(self.update_test_1_simulation_parameters)
        self.test_1_display_l_max_label.setBuddy(self.test_1_display_l_max)
        self.test_1_display_l_max_unit_label.setBuddy(self.test_1_display_l_max)

        self.test_1_display_l_min.setRange(self.display_l_min_min, self.display_l_min_max)
        self.test_1_display_l_min.setValue(self.display_l_min_default_value)
        self.test_1_display_l_min.valueChanged.connect(self.update_test_1_simulation_parameters)
        self.test_1_display_l_min_label.setBuddy(self.test_1_display_l_min)
        self.test_1_display_l_min_unit_label.setBuddy(self.test_1_display_l_min)

        self.test_1_display_contrast.setRange(self.display_contrast_min, self.display_contrast_max)
        self.test_1_display_contrast.setValue(self.display_contrast_default_value)
        self.test_1_display_contrast.valueChanged.connect(self.update_test_1_simulation_parameters)
        self.test_1_display_contrast_label.setBuddy(self.test_1_display_l_max)

        self.test_1_display_gamma.setRange(self.display_gamma_min, self.display_gamma_max)
        self.test_1_display_gamma.setValue(self.display_gamma_default_value)
        self.test_1_display_gamma.valueChanged.connect(self.update_test_1_simulation_parameters)
        self.test_1_display_gamma_label.setBuddy(self.test_1_display_l_max)

        self.test_1_display_reflectivity.setRange(self.display_reflectivity_min, self.display_reflectivity_max)
        self.test_1_display_reflectivity.setValue(self.display_reflectivity_default_value)
        self.test_1_display_reflectivity.valueChanged.connect(self.update_test_1_simulation_parameters)
        self.test_1_display_reflectivity_label.setBuddy(self.test_1_display_l_max)

        self.test_1_display_group_box_layout.addWidget(self.test_1_display_same_as_reference, 0, 0, 1, 3)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_l_max_label, 1, 0)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_l_max, 1, 1)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_l_max_unit_label, 1, 2)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_l_min_label, 2, 0)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_l_min, 2, 1)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_l_min_unit_label, 2, 2)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_contrast_label, 3, 0)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_contrast, 3, 1)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_gamma_label, 4, 0)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_gamma, 4, 1)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_reflectivity_label, 5, 0)
        self.test_1_display_group_box_layout.addWidget(self.test_1_display_reflectivity, 5, 1)

        self.test_1_display_group_box.setLayout(self.test_1_display_group_box_layout)

    def create_test_2_display_group_box(self):
        self.test_2_display_same_as_reference.setChecked(True)
        self.test_2_display_same_as_reference.stateChanged.connect(self.update_all_display)

        self.test_2_display_l_max.setRange(self.display_l_max_min, self.display_l_max_max)
        self.test_2_display_l_max.setValue(self.display_l_max_default_value)
        self.test_2_display_l_max.valueChanged.connect(self.update_test_2_simulation_parameters)
        self.test_2_display_l_max_label.setBuddy(self.test_2_display_l_max)
        self.test_2_display_l_max_unit_label.setBuddy(self.test_2_display_l_max)

        self.test_2_display_l_min.setRange(self.display_l_min_min, self.display_l_min_max)
        self.test_2_display_l_min.setValue(self.display_l_min_default_value)
        self.test_2_display_l_min.valueChanged.connect(self.update_test_2_simulation_parameters)
        self.test_2_display_l_min_label.setBuddy(self.test_2_display_l_min)
        self.test_2_display_l_min_unit_label.setBuddy(self.test_2_display_l_min)

        self.test_2_display_contrast.setRange(self.display_contrast_min, self.display_contrast_max)
        self.test_2_display_contrast.setValue(self.display_contrast_default_value)
        self.test_2_display_contrast.valueChanged.connect(self.update_test_2_simulation_parameters)
        self.test_2_display_contrast_label.setBuddy(self.test_2_display_l_max)

        self.test_2_display_gamma.setRange(self.display_gamma_min, self.display_gamma_max)
        self.test_2_display_gamma.setValue(self.display_gamma_default_value)
        self.test_2_display_gamma.valueChanged.connect(self.update_test_2_simulation_parameters)
        self.test_2_display_gamma_label.setBuddy(self.test_2_display_l_max)

        self.test_2_display_reflectivity.setRange(self.display_reflectivity_min, self.display_reflectivity_max)
        self.test_2_display_reflectivity.setValue(self.display_reflectivity_default_value)
        self.test_2_display_reflectivity.valueChanged.connect(self.update_test_2_simulation_parameters)
        self.test_2_display_reflectivity_label.setBuddy(self.test_2_display_l_max)

        self.test_2_display_group_box_layout.addWidget(self.test_2_display_same_as_reference, 0, 0, 1, 3)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_l_max_label, 1, 0)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_l_max, 1, 1)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_l_max_unit_label, 1, 2)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_l_min_label, 2, 0)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_l_min, 2, 1)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_l_min_unit_label, 2, 2)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_contrast_label, 3, 0)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_contrast, 3, 1)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_gamma_label, 4, 0)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_gamma, 4, 1)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_reflectivity_label, 5, 0)
        self.test_2_display_group_box_layout.addWidget(self.test_2_display_reflectivity, 5, 1)

        self.test_2_display_group_box.setLayout(self.test_2_display_group_box_layout)

    def load_test_1_image(self):
        self.test_1_image_view.open()
        self.image_test_1 = self.test_1_image_view.getDisplayedImage()

        self.window_test_1_image_view = Window("Test 1 image", self.test_1_image_view)
        self.window_test_1_image_view.show()

    def load_test_2_image(self):
        self.test_2_image_view.open()
        self.image_test_2 = self.test_2_image_view.getDisplayedImage()

        self.window_test_2_image_view = Window("Test 2 image", self.test_2_image_view)
        self.window_test_2_image_view.show()

    def create_test_1_image_group_box(self):
        self.button_load_test_1_image.clicked.connect(self.load_test_1_image)

        self.test_1_image_group_box_layout.addWidget(self.button_load_test_1_image)

        self.test_1_image_group_box.setLayout(self.test_1_image_group_box_layout)

    def create_test_2_image_group_box(self):
        self.button_load_test_2_image.clicked.connect(self.load_test_2_image)

        self.test_2_image_group_box_layout.addWidget(self.button_load_test_2_image)

        self.test_2_image_group_box.setLayout(self.test_2_image_group_box_layout)

    def create_test_1_evaluation_options_group_box(self):
        self.test_1_evaluation_options_same_as_reference.setChecked(False)
        self.test_1_evaluation_options_same_as_reference.stateChanged.connect(
            self.update_all_illumination_simulation_parameters)

        self.apply_reflection_test_1.setChecked(True)
        self.apply_reflection_test_1.stateChanged.connect(self.update_test_1_illumination_simulation_parameters)
        self.apply_screen_dimming_test_1.setChecked(True)
        self.apply_screen_dimming_test_1.stateChanged.connect(self.update_test_1_illumination_simulation_parameters)

        self.test_1_evaluation_options_group_box_layout.addWidget(self.test_1_evaluation_options_same_as_reference)
        self.test_1_evaluation_options_group_box_layout.addWidget(self.apply_reflection_test_1)
        self.test_1_evaluation_options_group_box_layout.addWidget(self.apply_screen_dimming_test_1)
        self.test_1_evaluation_options_group_box_layout.addStretch(1)

        self.test_1_evaluation_options_group_box.setLayout(self.test_1_evaluation_options_group_box_layout)

    def create_test_2_evaluation_options_group_box(self):
        self.test_2_evaluation_options_same_as_reference.setChecked(False)
        self.test_2_evaluation_options_same_as_reference.stateChanged.connect(
            self.update_all_illumination_simulation_parameters)

        self.apply_reflection_test_2.setChecked(True)
        self.apply_reflection_test_2.stateChanged.connect(self.update_test_2_illumination_simulation_parameters)
        self.apply_screen_dimming_test_2.setChecked(True)
        self.apply_screen_dimming_test_2.stateChanged.connect(self.update_test_2_illumination_simulation_parameters)

        self.test_2_evaluation_options_group_box_layout.addWidget(self.test_2_evaluation_options_same_as_reference)
        self.test_2_evaluation_options_group_box_layout.addWidget(self.apply_reflection_test_2)
        self.test_2_evaluation_options_group_box_layout.addWidget(self.apply_screen_dimming_test_2)
        self.test_2_evaluation_options_group_box_layout.addStretch(1)

        self.test_2_evaluation_options_group_box.setLayout(self.test_2_evaluation_options_group_box_layout)

    def create_test_1_group_box(self):
        self.create_test_1_illumination_condition_group_box()
        self.create_test_1_display_group_box()
        self.create_test_1_image_group_box()
        self.create_test_1_evaluation_options_group_box()

        self.test_1_group_box_layout.addWidget(self.test_1_illumination_condition_group_box)
        self.test_1_group_box_layout.addWidget(self.test_1_display_group_box)
        self.test_1_group_box_layout.addWidget(self.test_1_evaluation_options_group_box)
        self.test_1_group_box_layout.addWidget(self.test_1_image_group_box)
        self.test_1_group_box_layout.addStretch(1)

        self.test_1_group_box.setLayout(self.test_1_group_box_layout)

        self.update_test_1_simulation_parameters()

    def create_test_2_group_box(self):
        self.create_test_2_illumination_condition_group_box()
        self.create_test_2_display_group_box()
        self.create_test_2_image_group_box()
        self.create_test_2_evaluation_options_group_box()

        self.test_2_group_box_layout.addWidget(self.test_2_illumination_condition_group_box)
        self.test_2_group_box_layout.addWidget(self.test_2_display_group_box)
        self.test_2_group_box_layout.addWidget(self.test_2_evaluation_options_group_box)
        self.test_2_group_box_layout.addWidget(self.test_2_image_group_box)
        self.test_2_group_box_layout.addStretch(1)

        self.test_2_group_box.setLayout(self.test_2_group_box_layout)

        self.update_test_2_simulation_parameters()

    def create_evaluation_metrics_group_box(self):
        self.iqa_msssim.setChecked(True)
        self.iqa_msssim.stateChanged.connect(self.update_iqa_list)
        self.iqa_lpips.stateChanged.connect(self.update_iqa_list)
        self.iqa_vtamiq.stateChanged.connect(self.update_iqa_list)

        self.update_iqa_list()

        self.evaluation_metrics_group_box_layout.addWidget(self.iqa_msssim)
        self.evaluation_metrics_group_box_layout.addWidget(self.iqa_lpips)
        self.evaluation_metrics_group_box_layout.addWidget(self.iqa_vtamiq)
        self.evaluation_metrics_group_box_layout.addStretch(1)

        self.evaluation_metrics_group_box.setLayout(self.evaluation_metrics_group_box_layout)

    def create_evaluation_parameters_group_box(self):
        self.create_reference_group_box()
        self.create_test_1_group_box()
        self.create_test_2_group_box()
        self.create_evaluation_metrics_group_box()

        self.evaluation_parameters_group_box_layout.addWidget(self.reference_group_box, 0, 0)
        self.evaluation_parameters_group_box_layout.addWidget(self.test_1_group_box, 0, 1)
        self.evaluation_parameters_group_box_layout.addWidget(self.test_2_group_box, 0, 2)
        self.evaluation_parameters_group_box_layout.addWidget(self.evaluation_metrics_group_box, 1, 0, -1, -1)

        self.evaluation_parameters_group_box.setLayout(self.evaluation_parameters_group_box_layout)

    def create_simulated_reference_group_box(self):
        self.button_simulate_reference_image.clicked.connect(self.simulate_reference_image)

        self.simulated_reference_group_box_layout.addWidget(self.button_simulate_reference_image)

        self.simulated_reference_group_box.setLayout(self.simulated_reference_group_box_layout)

    def create_simulated_test_1_group_box(self):
        self.button_simulate_test_1_image.clicked.connect(self.simulate_test_1_image)

        self.simulated_test_1_group_box_layout.addWidget(self.button_simulate_test_1_image)
        self.simulated_test_1_group_box_layout.addStretch(1)

        self.simulated_test_1_group_box.setLayout(self.simulated_test_1_group_box_layout)

    def create_simulated_test_2_group_box(self):
        self.button_simulate_test_2_image.clicked.connect(self.simulate_test_2_image)

        self.simulated_test_2_group_box_layout.addWidget(self.button_simulate_test_2_image)
        self.simulated_test_2_group_box_layout.addStretch(1)

        self.simulated_test_2_group_box.setLayout(self.simulated_test_2_group_box_layout)

    def create_results_metrics_group_box(self):
        self.button_compute_iqa_metrics.clicked.connect(self.compute_iqa_metrics)

        self.output_metric_computation.moveCursor(QTextCursor.Start)
        self.output_metric_computation.ensureCursorVisible()
        self.output_metric_computation.setFontPointSize(20)
        self.output_metric_computation.setFontWeight(QFont.DemiBold)
        # self.output_metric_computation.setLineWrapMode(QTextEdit.FixedPixelWidth)

        self.results_metrics_group_box_layout.addWidget(self.button_compute_iqa_metrics)
        self.results_metrics_group_box_layout.addWidget(self.output_metric_computation)

        self.results_metrics_group_box.setLayout(self.results_metrics_group_box_layout)

    def update_iqa_list(self):
        self.iqa_list = []
        if self.iqa_msssim.isChecked():
            self.iqa_list.append(iqa_msssim)
        if self.iqa_lpips.isChecked():
            self.iqa_list.append(iqa_lpips)
        if self.iqa_vtamiq.isChecked():
            self.iqa_list.append(iqa_vtamiq)

    def compute_iqa_metrics(self):
        sim_params_reference = new_simul_params(illuminant=self.reference_illumination_condition.value(),
                                                apply_reflection=self.apply_reflection_reference.isChecked(),
                                                apply_screen_dimming=self.apply_screen_dimming_reference.isChecked(),
                                                use_luminance_only=False)
        sim_params_test_1 = new_simul_params(illuminant=self.test_1_illumination_condition.value(),
                                             apply_reflection=self.apply_reflection_test_1.isChecked(),
                                             apply_screen_dimming=self.apply_screen_dimming_test_1.isChecked(),
                                             use_luminance_only=False)
        sim_params_test_2 = new_simul_params(illuminant=self.test_2_illumination_condition.value(),
                                             apply_reflection=self.apply_reflection_test_2.isChecked(),
                                             apply_screen_dimming=self.apply_screen_dimming_test_2.isChecked(),
                                             use_luminance_only=False)

        iqa_tool = IqaTool(iqa_variants=self.iqa_list, use_age_simulation=self.apply_age_simulation.isChecked())

        iqa_qr_1 = iqa_tool.compute_iqa_custom(rgb_view(self.image_reference).astype(float) / 255.,
                                               rgb_view(self.image_test_1).astype(float) / 255.,
                                               sim_params_reference,
                                               sim_params_test_1,
                                               observer_age=self.age_simulation_age_value.value(),
                                               dm1=self.dm_ref,
                                               dm2=self.dm_test_1
                                               )

        iqa_qr_2 = iqa_tool.compute_iqa_custom(rgb_view(self.image_reference).astype(float) / 255.,
                                               rgb_view(self.image_test_2).astype(float) / 255.,
                                               sim_params_reference,
                                               sim_params_test_2,
                                               observer_age=self.age_simulation.value(),
                                               dm1=self.dm_ref,
                                               dm2=self.dm_test_2
                                               )

        print('###############')
        print("IQA results:")
        for iqa_variant in self.iqa_list:
            iqa_variant_name = self.metric_name_dict[iqa_variant.name]
            print(iqa_variant_name, 'Reference vs Test 1 - ', iqa_qr_1[iqa_variant.name])
            print(iqa_variant_name, 'Reference vs Test 2 - ', iqa_qr_2[iqa_variant.name])
            # compute and display preference for VTAMIQ
            if iqa_variant.name == iqa_vtamiq.name:
                pref = 100 * compute_pairwise_pref(iqa_qr_1[iqa_variant.name], iqa_qr_2[iqa_variant.name])
                if pref > 50:
                    print(iqa_variant_name, 'Test 1 is preferred with', float2str3(pref), '% probability')
                else:
                    print(iqa_variant_name, 'Test 2 is preferred with', float2str3(100. - pref), '% probability')

        print('Done.\n')

    def update_reference_simulation_parameters(self):
        self.update_reference_display()
        self.reference_target_screen_brightness.setNum(
            self.dm_ref.get_L_max(self.reference_illumination_condition.value(),
                                  self.apply_screen_dimming_reference.isChecked()))

    def update_reference_display(self):
        self.dm_ref = DisplayModel(L_max=self.reference_display_l_max.value(),
                                   L_min=self.reference_display_l_min.value(),
                                   L_contrast_ratio=self.reference_display_contrast.value(),
                                   gamma=self.reference_display_gamma.value(),
                                   reflectivity=self.reference_display_reflectivity.value())
        self.ddm_ref = DisplayDegradationModel(display_model=self.dm_ref)

    def update_all_illumination_simulation_parameters(self):
        self.reference_target_screen_brightness.setNum(
            self.dm_ref.get_L_max(self.reference_illumination_condition.value(),
                                  self.apply_screen_dimming_reference.isChecked()))
        self.update_test_1_illumination_simulation_parameters()
        self.update_test_2_illumination_simulation_parameters()

    def update_age_simulation_parameters(self):
        # no need to do anything here
        pass

    def update_test_1_simulation_parameters(self):
        self.update_test_1_illumination_simulation_parameters()
        self.update_test_1_display()
        self.test_1_target_screen_brightness.setNum(self.dm_test_1.get_L_max(self.test_1_illumination_condition.value(),
                                                                             self.apply_screen_dimming_test_1.isChecked()))

    def update_test_2_simulation_parameters(self):
        self.update_test_2_illumination_simulation_parameters()
        self.update_test_2_display()
        self.test_2_target_screen_brightness.setNum(self.dm_test_2.get_L_max(self.test_2_illumination_condition.value(),
                                                                             self.apply_screen_dimming_test_2.isChecked()))

    def update_test_1_illumination_simulation_parameters(self):
        if self.test_1_illumination_same_as_reference.isChecked():
            self.test_1_illumination_condition.setValue(self.reference_illumination_condition.value())
            self.test_1_illumination_condition.setDisabled(True)
        else:
            self.test_1_illumination_condition.setDisabled(False)

        if self.test_1_evaluation_options_same_as_reference.isChecked():
            self.apply_reflection_test_1.setChecked(self.apply_reflection_reference.isChecked())
            self.apply_screen_dimming_test_1.setChecked(self.apply_screen_dimming_reference.isChecked())
            self.apply_reflection_test_1.setDisabled(True)
            self.apply_screen_dimming_test_1.setDisabled(True)
        else:
            self.apply_reflection_test_1.setDisabled(False)
            self.apply_screen_dimming_test_1.setDisabled(False)

        if self.dm_test_1 is not None:
            self.test_1_target_screen_brightness.setNum(
                self.dm_test_1.get_L_max(self.test_1_illumination_condition.value(),
                                         self.apply_screen_dimming_test_1.isChecked()))

    def update_test_2_illumination_simulation_parameters(self):
        if self.test_2_illumination_same_as_reference.isChecked():
            self.test_2_illumination_condition.setValue(self.reference_illumination_condition.value())
            self.test_2_illumination_condition.setDisabled(True)
        else:
            self.test_2_illumination_condition.setDisabled(False)

        if self.test_2_evaluation_options_same_as_reference.isChecked():
            self.apply_reflection_test_2.setChecked(self.apply_reflection_reference.isChecked())
            self.apply_screen_dimming_test_2.setChecked(self.apply_screen_dimming_reference.isChecked())
            self.apply_reflection_test_2.setDisabled(True)
            self.apply_screen_dimming_test_2.setDisabled(True)
        else:
            self.apply_reflection_test_2.setDisabled(False)
            self.apply_screen_dimming_test_2.setDisabled(False)
        if self.dm_test_2 is not None:
            self.test_2_target_screen_brightness.setNum(
                self.dm_test_2.get_L_max(self.test_2_illumination_condition.value(),
                                         self.apply_screen_dimming_test_2.isChecked()))

    def update_test_1_display(self):
        if self.test_1_display_same_as_reference.isChecked():
            self.test_1_display_l_max.setValue(self.reference_display_l_max.value())
            self.test_1_display_l_min.setValue(self.reference_display_l_min.value())
            self.test_1_display_contrast.setValue(self.reference_display_contrast.value())
            self.test_1_display_gamma.setValue(self.reference_display_gamma.value())
            self.test_1_display_reflectivity.setValue(self.reference_display_reflectivity.value())
            self.test_1_display_l_max.setDisabled(True)
            self.test_1_display_l_min.setDisabled(True)
            self.test_1_display_contrast.setDisabled(True)
            self.test_1_display_gamma.setDisabled(True)
            self.test_1_display_reflectivity.setDisabled(True)
        else:
            self.test_1_display_l_max.setDisabled(False)
            self.test_1_display_l_min.setDisabled(False)
            self.test_1_display_contrast.setDisabled(False)
            self.test_1_display_gamma.setDisabled(False)
            self.test_1_display_reflectivity.setDisabled(False)

        self.dm_test_1 = DisplayModel(L_max=self.test_1_display_l_max.value(),
                                      L_min=self.test_1_display_l_min.value(),
                                      L_contrast_ratio=self.test_1_display_contrast.value(),
                                      gamma=self.test_1_display_gamma.value(),
                                      reflectivity=self.test_1_display_reflectivity.value())
        self.ddm_test_1 = DisplayDegradationModel(display_model=self.dm_test_1)

    def update_test_2_display(self):
        if self.test_2_display_same_as_reference.isChecked():
            self.test_2_display_l_max.setValue(self.reference_display_l_max.value())
            self.test_2_display_l_min.setValue(self.reference_display_l_min.value())
            self.test_2_display_contrast.setValue(self.reference_display_contrast.value())
            self.test_2_display_gamma.setValue(self.reference_display_gamma.value())
            self.test_2_display_reflectivity.setValue(self.reference_display_reflectivity.value())
            self.test_2_display_l_max.setDisabled(True)
            self.test_2_display_l_min.setDisabled(True)
            self.test_2_display_contrast.setDisabled(True)
            self.test_2_display_gamma.setDisabled(True)
            self.test_2_display_reflectivity.setDisabled(True)
        else:
            self.test_2_display_l_max.setDisabled(False)
            self.test_2_display_l_min.setDisabled(False)
            self.test_2_display_contrast.setDisabled(False)
            self.test_2_display_gamma.setDisabled(False)
            self.test_2_display_reflectivity.setDisabled(False)

        self.dm_test_2 = DisplayModel(L_max=self.test_2_display_l_max.value(),
                                      L_min=self.test_2_display_l_min.value(),
                                      L_contrast_ratio=self.test_2_display_contrast.value(),
                                      gamma=self.test_2_display_gamma.value(),
                                      reflectivity=self.test_2_display_reflectivity.value())
        self.ddm_test_2 = DisplayDegradationModel(display_model=self.dm_test_2)

    def update_all_display(self):
        self.update_reference_display()
        self.update_test_1_display()
        self.update_test_2_display()

    def simulate_reference_image(self):
        image = imread_unknown_extension(self.reference_image_view.get_basename_no_ext(),
                                         self.reference_image_view.get_dir(), format_float=True)
        sim_params_reference = new_simul_params(illuminant=self.reference_illumination_condition.value(),
                                                apply_reflection=self.apply_reflection_reference.isChecked(),
                                                apply_screen_dimming=self.apply_screen_dimming_reference.isChecked(),
                                                use_luminance_only=False)

        simulated_reference_image = self.ddm_ref.simulate_displays_rgb(image, sim_params_reference)
        reference_file_name = 'simulated_reference_image_{}.jpeg'.format(self.simulated_reference_image_number)
        imwrite(reference_file_name, '.', simulated_reference_image)

        self.simulated_reference_image_view.open(reference_file_name)

        self.window_simulated_reference_image_view = Window(
            "Simulated reference image {}".format(self.simulated_reference_image_number),
            self.simulated_reference_image_view)
        self.window_simulated_reference_image_view.show()
        self.simulated_reference_image_number += 1

    def simulate_test_1_image(self):
        image = imread_unknown_extension(self.test_1_image_view.get_basename_no_ext(),
                                         self.test_1_image_view.get_dir(), format_float=True)
        sim_params_test = new_simul_params(illuminant=self.test_1_illumination_condition.value(),
                                           apply_reflection=self.apply_reflection_test_1.isChecked(),
                                           apply_screen_dimming=self.apply_screen_dimming_test_1.isChecked(),
                                           use_luminance_only=False)

        simulated_test_image = self.ddm_test_1.simulate_displays_rgb(image, sim_params_test)
        test_file_name = 'simulated_test_1_image_{}.jpeg'.format(self.simulated_test_1_image_number)
        imwrite(test_file_name, '.', simulated_test_image)

        self.simulated_test_1_image_view.open(test_file_name)

        self.window_simulated_test_1_image_view = Window(
            "Simulated test 1 image {}".format(self.simulated_test_1_image_number), self.simulated_test_1_image_view)
        self.window_simulated_test_1_image_view.show()
        self.simulated_test_1_image_number += 1

    def simulate_test_2_image(self):
        image = imread_unknown_extension(self.test_2_image_view.get_basename_no_ext(),
                                         self.test_2_image_view.get_dir(), format_float=True)
        sim_params_test = new_simul_params(illuminant=self.test_2_illumination_condition.value(),
                                           apply_reflection=self.apply_reflection_test_2.isChecked(),
                                           apply_screen_dimming=self.apply_screen_dimming_test_2.isChecked(),
                                           use_luminance_only=False)

        simulated_test_image = self.ddm_test_2.simulate_displays_rgb(image, sim_params_test)
        test_file_name = 'simulated_test_2_image_{}.jpeg'.format(self.simulated_test_2_image_number)
        imwrite(test_file_name, '.', simulated_test_image)

        self.simulated_test_2_image_view.open(test_file_name)

        self.window_simulated_test_2_image_view = Window(
            "Simulated test 2 image {}".format(self.simulated_test_2_image_number), self.simulated_test_2_image_view)
        self.window_simulated_test_2_image_view.show()
        self.simulated_test_2_image_number += 1

    def create_simulation_results_group_box(self):

        self.create_results_metrics_group_box()
        self.create_simulated_image_group_box()

        self.simulation_results_group_box_layout.addWidget(self.results_metrics_group_box)
        self.simulation_results_group_box_layout.addWidget(self.simulated_images_group_box)

        self.simulation_results_group_box.setLayout(self.simulation_results_group_box_layout)

    def create_simulated_image_group_box(self):
        self.create_simulated_reference_group_box()
        self.create_simulated_test_1_group_box()
        self.create_simulated_test_2_group_box()

        self.simulated_images_group_box_layout.addWidget(self.simulated_reference_group_box)
        self.simulated_images_group_box_layout.addWidget(self.simulated_test_1_group_box)
        self.simulated_images_group_box_layout.addWidget(self.simulated_test_2_group_box)

        self.simulated_images_group_box.setLayout(self.simulated_images_group_box_layout)


def set_dark_mode(app):
    app.setStyle("Fusion")

    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.Active, QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, Qt.darkGray)
    dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, Qt.darkGray)
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, Qt.darkGray)
    dark_palette.setColor(QPalette.Disabled, QPalette.Light, QColor(53, 53, 53))
    app.setPalette(dark_palette)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    set_dark_mode(app)

    ex = App()
    sys.exit(app.exec_())
