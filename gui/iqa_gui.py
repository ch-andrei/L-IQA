from utils.image_processing.image_tools import *
from DisplayModels.display_model import DisplayModel
from iqa_metrics.iqa_tool import new_simul_params, IqaTool
from iqa_metrics.metrics import (iqa_psnr, iqa_msssim, iqa_lpips, iqa_hdr_vdp)
import numpy as np
import os
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QTextCursor
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit, QDial, QDialog,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                             QPushButton, QRadioButton, QScrollBar, QSizePolicy, QSlider, QSpinBox,
                             QStyleFactory, QTableWidget, QTabWidget, QTextEdit, QVBoxLayout,
                             QWidget, QFileDialog)
from gui.qimageviewer import QImageViewer
from qimage2ndarray import array2qimage, rgb_view
import sys


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PIQA GUI'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        mainLayout = QGridLayout()

        self.createEvaluationParamatersGroupBox()
        self.createSimulationResultsGroupBox()

        mainLayout.addWidget(self.evaluationParamatersGroupBox, 0, 0)
        mainLayout.addWidget(self.simulationResultsGroupBox, 1, 0)

        sys.stdout = Stream(newText=self.onUpdateText)

        self.setLayout(mainLayout)
        self.show()

    def onUpdateText(self, text):
        cursor = self.outputMetricComputation.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.outputMetricComputation.setTextCursor(cursor)
        self.outputMetricComputation.ensureCursorVisible()

    def createReferenceIlluminationConditionGroupBox(self):
        self.referenceIlluminationConditionGroupBox = QGroupBox("Reference Illumination Condition")

        self.referenceIlluminationCondition = QSpinBox(self.referenceIlluminationConditionGroupBox)
        self.referenceIlluminationCondition.setRange(0, 30000)
        self.referenceIlluminationCondition.setValue(2000)
        self.referenceIlluminationCondition.valueChanged.connect(
            self.updateAllIlluminationSimulationParameters)
        self.referenceIlluminationLabel = QLabel("Illumination")
        self.referenceIlluminationLabel.setBuddy(self.referenceIlluminationCondition)
        self.referenceIlluminationUnitLabel = QLabel("lux")
        self.referenceIlluminationUnitLabel.setBuddy(self.referenceIlluminationCondition)

        self.referenceIlluminationConditionGroupBoxLayout = QGridLayout()
        self.referenceIlluminationConditionGroupBoxLayout.addWidget(
            self.referenceIlluminationLabel, 0, 0)
        self.referenceIlluminationConditionGroupBoxLayout.addWidget(
            self.referenceIlluminationCondition, 0, 1)
        self.referenceIlluminationConditionGroupBoxLayout.addWidget(
            self.referenceIlluminationUnitLabel, 0, 2)

        self.referenceIlluminationConditionGroupBox.setLayout(
            self.referenceIlluminationConditionGroupBoxLayout)

    def createReferenceDisplayGroupBox(self):
        self.referenceDisplayGroupBox = QGroupBox("Reference Display")

        self.referenceDisplayLmax = QSpinBox(self.referenceDisplayGroupBox)
        self.referenceDisplayLmax.setRange(0, 50000)
        self.referenceDisplayLmax.setValue(900)
        self.referenceDisplayLmax.valueChanged.connect(self.updateAllDisplay)
        self.referenceDisplayLmaxLabel = QLabel("Lmax")
        self.referenceDisplayLmaxLabel.setBuddy(self.referenceDisplayLmax)
        self.referenceDisplayLmaxUnitLabel = QLabel("cd/m2")
        self.referenceDisplayLmaxUnitLabel.setBuddy(self.referenceDisplayLmax)

        self.referenceDisplayContrast = QSpinBox(self.referenceDisplayGroupBox)
        self.referenceDisplayContrast.setRange(1, 100000)
        self.referenceDisplayContrast.setValue(800)
        self.referenceDisplayContrast.valueChanged.connect(self.updateAllDisplay)
        self.referenceDisplayContrastLabel = QLabel("Contrast")
        self.referenceDisplayContrastLabel.setBuddy(self.referenceDisplayLmax)

        self.referenceDisplayGroupBoxLayout = QGridLayout()
        self.referenceDisplayGroupBoxLayout.addWidget(self.referenceDisplayLmaxLabel, 0, 0)
        self.referenceDisplayGroupBoxLayout.addWidget(self.referenceDisplayLmax, 0, 1)
        self.referenceDisplayGroupBoxLayout.addWidget(self.referenceDisplayLmaxUnitLabel, 0, 2)
        self.referenceDisplayGroupBoxLayout.addWidget(self.referenceDisplayContrastLabel, 1, 0)
        self.referenceDisplayGroupBoxLayout.addWidget(self.referenceDisplayContrast, 1, 1)

        self.referenceDisplayGroupBox.setLayout(self.referenceDisplayGroupBoxLayout)

    def loadReferenceImage(self):
        self.referenceImageView.open()
        self.imageReference = self.referenceImageView.getDisplayedImage()

    def createReferenceImageGroupBox(self):
        self.referenceImageGroupBox = QGroupBox("Reference Image")

        self.referenceImageView = QImageViewer()
        self.buttonLoadReferenceImage = QPushButton('Load Image')
        self.buttonLoadReferenceImage.clicked.connect(self.loadReferenceImage)

        self.referenceImageGroupBoxLayout = QVBoxLayout()
        self.referenceImageGroupBoxLayout.addWidget(self.buttonLoadReferenceImage)
        self.referenceImageGroupBoxLayout.addWidget(self.referenceImageView)
        self.referenceImageGroupBoxLayout.addStretch(1)

        self.referenceImageGroupBox.setLayout(self.referenceImageGroupBoxLayout)

    def createReferenceEvaluationOptionsGroupBox(self):
        self.referenceEvaluationOptionsGroupBox = QGroupBox("Reference Evaluation Options")

        self.useLuminanceOnlyReference = QCheckBox("Use luminance only")
        self.useLuminanceOnlyReference.setChecked(True)
        self.useLuminanceOnlyReference.stateChanged.connect(
            self.updateAllIlluminationSimulationParameters)
        self.applyReflectionReference = QCheckBox("Apply reflection")
        self.applyReflectionReference.setChecked(True)
        self.applyReflectionReference.stateChanged.connect(
            self.updateAllIlluminationSimulationParameters)
        self.applyScreenDimingReference = QCheckBox("Apply screen diming")
        self.applyScreenDimingReference.setChecked(True)
        self.applyScreenDimingReference.stateChanged.connect(
            self.updateAllIlluminationSimulationParameters)

        self.referenceEvaluationOptionsGroupBoxLayout = QVBoxLayout()
        self.referenceEvaluationOptionsGroupBoxLayout.addWidget(self.useLuminanceOnlyReference)
        self.referenceEvaluationOptionsGroupBoxLayout.addWidget(self.applyReflectionReference)
        self.referenceEvaluationOptionsGroupBoxLayout.addWidget(self.applyScreenDimingReference)
        self.referenceEvaluationOptionsGroupBoxLayout.addStretch(1)

        self.referenceEvaluationOptionsGroupBox.setLayout(
            self.referenceEvaluationOptionsGroupBoxLayout)

    def createReferenceGroupBox(self):
        self.referenceGroupBox = QGroupBox("Reference Parameters")

        self.createReferenceIlluminationConditionGroupBox()
        self.createReferenceDisplayGroupBox()
        self.createReferenceImageGroupBox()
        self.createReferenceEvaluationOptionsGroupBox()

        self.referenceGroupBoxLayout = QVBoxLayout()
        self.referenceGroupBoxLayout.addWidget(self.referenceIlluminationConditionGroupBox)
        self.referenceGroupBoxLayout.addWidget(self.referenceDisplayGroupBox)
        self.referenceGroupBoxLayout.addWidget(self.referenceEvaluationOptionsGroupBox)
        self.referenceGroupBoxLayout.addWidget(self.referenceImageGroupBox)
        self.referenceGroupBoxLayout.addStretch(1)

        self.referenceGroupBox.setLayout(self.referenceGroupBoxLayout)

        self.updateReferenceSimulationParameters();

    def createTestIlluminationConditionGroupBox(self):
        self.testIlluminationConditionGroupBox = QGroupBox("Test Illumination Condition")

        self.testIlluminationCondition = QSpinBox(self.testIlluminationConditionGroupBox)
        self.testIlluminationCondition.setRange(0, 30000)
        self.testIlluminationCondition.setValue(1000)
        self.testIlluminationCondition.valueChanged.connect(
            self.updateTestIlluminationSimulationParameters)
        self.testIlluminationLabel = QLabel("Illumination")
        self.testIlluminationLabel.setBuddy(self.testIlluminationCondition)
        self.testIlluminationUnitLabel = QLabel("lux")
        self.testIlluminationUnitLabel.setBuddy(self.testIlluminationCondition)
        self.testIlluminationSameAsReference = QCheckBox("Use same as Reference")
        self.testIlluminationSameAsReference.setChecked(True)
        self.testIlluminationSameAsReference.stateChanged.connect(
            self.updateAllIlluminationSimulationParameters)

        self.testIlluminationConditionGroupBoxLayout = QGridLayout()
        self.testIlluminationConditionGroupBoxLayout.addWidget(self.testIlluminationSameAsReference, 0, 0)
        self.testIlluminationConditionGroupBoxLayout.addWidget(self.testIlluminationLabel, 1, 0)
        self.testIlluminationConditionGroupBoxLayout.addWidget(self.testIlluminationCondition, 1, 1)
        self.testIlluminationConditionGroupBoxLayout.addWidget(self.testIlluminationUnitLabel, 1, 2)

        self.testIlluminationConditionGroupBox.setLayout(
            self.testIlluminationConditionGroupBoxLayout)

    def createTestDisplayGroupBox(self):
        self.testDisplayGroupBox = QGroupBox("Test Display")

        self.testDisplaySameAsReference = QCheckBox("Use same as Reference")
        self.testDisplaySameAsReference.setChecked(True)
        self.testDisplaySameAsReference.stateChanged.connect(self.updateAllDisplay)

        self.testDisplayLmax = QSpinBox(self.testDisplayGroupBox)
        self.testDisplayLmax.setRange(0, 50000)
        self.testDisplayLmax.setValue(500)
        self.testDisplayLmax.valueChanged.connect(self.updateTestDisplay)
        self.testDisplayLmaxLabel = QLabel("Lmax")
        self.testDisplayLmaxLabel.setBuddy(self.testDisplayLmax)
        self.testDisplayLmaxUnitLabel = QLabel("cd/m2")
        self.testDisplayLmaxUnitLabel.setBuddy(self.testDisplayLmax)

        self.testDisplayContrast = QSpinBox(self.testDisplayGroupBox)
        self.testDisplayContrast.setRange(1, 100000)
        self.testDisplayContrast.setValue(1000)
        self.testDisplayContrast.valueChanged.connect(self.updateTestDisplay)
        self.testDisplayContrastLabel = QLabel("Contrast")
        self.testDisplayContrastLabel.setBuddy(self.testDisplayLmax)

        self.testDisplayGroupBoxLayout = QGridLayout()
        self.testDisplayGroupBoxLayout.addWidget(self.testDisplaySameAsReference, 0, 0)
        self.testDisplayGroupBoxLayout.addWidget(self.testDisplayLmaxLabel, 1, 0)
        self.testDisplayGroupBoxLayout.addWidget(self.testDisplayLmax, 1, 1)
        self.testDisplayGroupBoxLayout.addWidget(self.testDisplayLmaxUnitLabel, 1, 2)
        self.testDisplayGroupBoxLayout.addWidget(self.testDisplayContrastLabel, 2, 0)
        self.testDisplayGroupBoxLayout.addWidget(self.testDisplayContrast, 2, 1)

        self.testDisplayGroupBox.setLayout(self.testDisplayGroupBoxLayout)

    def loadTestImage(self):
        self.testImageView.open()
        self.imageTest = self.testImageView.getDisplayedImage()

    def createTestImageGroupBox(self):
        self.testImageGroupBox = QGroupBox("Test Image")

        self.testImageView = QImageViewer()
        self.buttonLoadTestImage = QPushButton('Load Image')
        self.buttonLoadTestImage.clicked.connect(self.loadTestImage)

        self.testImageGroupBoxLayout = QVBoxLayout()
        self.testImageGroupBoxLayout.addWidget(self.buttonLoadTestImage)
        self.testImageGroupBoxLayout.addWidget(self.testImageView)
        self.testImageGroupBoxLayout.addStretch(1)

        self.testImageGroupBox.setLayout(self.testImageGroupBoxLayout)

    def createTestEvaluationOptionsGroupBox(self):
        self.testEvaluationOptionsGroupBox = QGroupBox("Test Evaluation Options")

        self.testEvaluationOptionsSameAsReference = QCheckBox("Use same as Reference")
        self.testEvaluationOptionsSameAsReference.setChecked(True)
        self.testEvaluationOptionsSameAsReference.stateChanged.connect(self.updateAllIlluminationSimulationParameters)

        self.useLuminanceOnlyTest = QCheckBox("Use luminance only")
        self.useLuminanceOnlyTest.setChecked(True)
        self.useLuminanceOnlyTest.stateChanged.connect(
            self.updateTestIlluminationSimulationParameters)
        self.applyReflectionTest = QCheckBox("Apply reflection")
        self.applyReflectionTest.setChecked(True)
        self.applyReflectionTest.stateChanged.connect(
            self.updateTestIlluminationSimulationParameters)
        self.applyScreenDimingTest = QCheckBox("Apply screen diming")
        self.applyScreenDimingTest.setChecked(True)
        self.applyScreenDimingTest.stateChanged.connect(
            self.updateTestIlluminationSimulationParameters)

        self.testEvaluationOptionsGroupBoxLayout = QVBoxLayout()
        self.testEvaluationOptionsGroupBoxLayout.addWidget(self.testEvaluationOptionsSameAsReference)
        self.testEvaluationOptionsGroupBoxLayout.addWidget(self.useLuminanceOnlyTest)
        self.testEvaluationOptionsGroupBoxLayout.addWidget(self.applyReflectionTest)
        self.testEvaluationOptionsGroupBoxLayout.addWidget(self.applyScreenDimingTest)
        self.testEvaluationOptionsGroupBoxLayout.addStretch(1)

        self.testEvaluationOptionsGroupBox.setLayout(self.testEvaluationOptionsGroupBoxLayout)

    def createTestGroupBox(self):
        self.testGroupBox = QGroupBox("Test Parameters")

        self.createTestIlluminationConditionGroupBox()
        self.createTestDisplayGroupBox()
        self.createTestImageGroupBox()
        self.createTestEvaluationOptionsGroupBox()

        self.testGroupBoxLayout = QVBoxLayout()
        self.testGroupBoxLayout.addWidget(self.testIlluminationConditionGroupBox)
        self.testGroupBoxLayout.addWidget(self.testDisplayGroupBox)
        self.testGroupBoxLayout.addWidget(self.testEvaluationOptionsGroupBox)
        self.testGroupBoxLayout.addWidget(self.testImageGroupBox)
        self.testGroupBoxLayout.addStretch(1)

        self.testGroupBox.setLayout(self.testGroupBoxLayout)

        self.updateTestSimulationParameters()

    def createEvaluationMetricsGroupBox(self):
        self.evaluationMetricsGroupBox = QGroupBox("Evaluation metrics")

        self.IQA_PSNR = QCheckBox("PSNR")
        self.IQA_PSNR.stateChanged.connect(self.updateIQAList)
        self.IQA_MSSSIM = QCheckBox("MSSSIM")
        self.IQA_MSSSIM.setChecked(True)
        self.IQA_MSSSIM.stateChanged.connect(self.updateIQAList)
        self.IQA_LPIPS = QCheckBox("LPIPS")
        self.IQA_LPIPS.stateChanged.connect(self.updateIQAList)
        self.IQA_HDR_VDP = QCheckBox("HDR-VDP")
        self.IQA_HDR_VDP.stateChanged.connect(self.updateIQAList)

        self.updateIQAList()

        self.evaluationMetricsGroupBoxLayout = QHBoxLayout()
        self.evaluationMetricsGroupBoxLayout.addWidget(self.IQA_PSNR)
        self.evaluationMetricsGroupBoxLayout.addWidget(self.IQA_MSSSIM)
        self.evaluationMetricsGroupBoxLayout.addWidget(self.IQA_LPIPS)
        self.evaluationMetricsGroupBoxLayout.addWidget(self.IQA_HDR_VDP)
        self.evaluationMetricsGroupBoxLayout.addStretch(1)

        self.evaluationMetricsGroupBox.setLayout(self.evaluationMetricsGroupBoxLayout)

    def createEvaluationParamatersGroupBox(self):
        self.evaluationParamatersGroupBox = QGroupBox("Evaluation Parameters")

        self.evaluationParamatersGroupBoxLayout = QGridLayout()

        self.createReferenceGroupBox()
        self.createTestGroupBox()
        self.createEvaluationMetricsGroupBox()

        self.evaluationParamatersGroupBoxLayout.addWidget(self.referenceGroupBox, 0, 0)
        self.evaluationParamatersGroupBoxLayout.addWidget(self.testGroupBox, 0, 1)
        self.evaluationParamatersGroupBoxLayout.addWidget(self.evaluationMetricsGroupBox, 1, 0, -1, -1)

        self.evaluationParamatersGroupBox.setLayout(self.evaluationParamatersGroupBoxLayout)

    def createSimulatedReferenceGroupBox(self):
        self.simulatedReferenceGroupBox = QGroupBox("Simulated Reference Image")

        self.simulatedReferenceImageView = QImageViewer()
        self.buttonSimulateReferenceImage = QPushButton('Simulate Reference Image')
        self.buttonSimulateReferenceImage.clicked.connect(self.simulateReferenceImage)

        self.simulatedReferenceGroupBoxLayout = QVBoxLayout()
        self.simulatedReferenceGroupBoxLayout.addWidget(self.buttonSimulateReferenceImage)
        self.simulatedReferenceGroupBoxLayout.addWidget(self.simulatedReferenceImageView)
        self.simulatedReferenceGroupBoxLayout.addStretch(1)

        self.simulatedReferenceGroupBox.setLayout(self.simulatedReferenceGroupBoxLayout)

    def createSimulatedTestGroupBox(self):
        self.simulatedTestGroupBox = QGroupBox("Simulated Test Image")

        self.simulatedTestImageView = QImageViewer()
        self.buttonSimulateTestImage = QPushButton('Simulate Test Image')
        self.buttonSimulateTestImage.clicked.connect(self.simulateTestImage)

        self.simulatedTestGroupBoxLayout = QVBoxLayout()
        self.simulatedTestGroupBoxLayout.addWidget(self.buttonSimulateTestImage)
        self.simulatedTestGroupBoxLayout.addWidget(self.simulatedTestImageView)
        self.simulatedTestGroupBoxLayout.addStretch(1)

        self.simulatedTestGroupBox.setLayout(self.simulatedTestGroupBoxLayout)

    def createResultsMetricsGroupBox(self):
        self.resultsMetricsGroupBox = QGroupBox("Simulated Test Image")

        self.simulatedTestImage = QImageViewer()
        self.buttonComputeIQAMetrics = QPushButton('Compute IQA metric(s0)')
        self.buttonComputeIQAMetrics.clicked.connect(self.computeIQAMetrics)

        self.outputMetricComputation = QTextEdit()
        self.outputMetricComputation.moveCursor(QTextCursor.Start)
        self.outputMetricComputation.ensureCursorVisible()
        self.outputMetricComputation.setLineWrapColumnOrWidth(500)
        self.outputMetricComputation.setLineWrapMode(QTextEdit.FixedPixelWidth)

        self.resultsMetricsGroupBoxLayout = QVBoxLayout()
        self.resultsMetricsGroupBoxLayout.addWidget(self.buttonComputeIQAMetrics)
        self.resultsMetricsGroupBoxLayout.addWidget(self.outputMetricComputation)
        self.resultsMetricsGroupBoxLayout.addStretch(1)

        self.resultsMetricsGroupBox.setLayout(self.resultsMetricsGroupBoxLayout)

    def updateIQAList(self):
        self.iqa_list = []
        if (self.IQA_PSNR.isChecked()):
            self.iqa_list.append(iqa_psnr)
        if (self.IQA_MSSSIM.isChecked()):
            self.iqa_list.append(iqa_msssim)
        if (self.IQA_LPIPS.isChecked()):
            self.iqa_list.append(iqa_lpips)
        if (self.IQA_HDR_VDP.isChecked()):
            self.iqa_list.append(iqa_hdr_vdp)

    def computeIQAMetrics(self):
        iqa_tool = IqaTool(iqa_to_use=self.iqa_list, tmqi_use_original=False)

        iqa_qr = iqa_tool.compute_iqa_custom(rgb_view(self.imageReference).astype(np.float) / 255.,
                                             rgb_view(self.imageTest).astype(np.float) / 255.,
                                             self.simParamsReference,
                                             self.simParamsTest,
                                             dm1=self.ddmRef, dm2=self.ddmTest)
        print('###############')
        print(iqa_qr)
        print('\n')

    def updateReferenceSimulationParameters(self):
        self.updateReferenceIlluminationSimulationParameters()
        self.updateReferenceDisplay()

    def updateReferenceIlluminationSimulationParameters(self):
        self.simParamsReference = new_simul_params(illuminant=self.referenceIlluminationCondition.value(),
                                                 use_luminance_only=self.useLuminanceOnlyReference.isChecked(),
                                                 apply_reflection=self.applyReflectionReference.isChecked(),
                                                 apply_screen_dimming=self.applyScreenDimingReference.isChecked())

    def updateReferenceDisplay(self):
        self.ddmRef = DisplayModel(L_max=self.referenceDisplayLmax.value(),
                                   contrast_ratio=self.referenceDisplayContrast.value())

    def updateAllIlluminationSimulationParameters(self):
        self.updateReferenceIlluminationSimulationParameters()
        self.updateTestIlluminationSimulationParameters()

    def updateTestSimulationParameters(self):
        self.updateTestIlluminationSimulationParameters()
        self.updateTestDisplay()

    def updateTestIlluminationSimulationParameters(self):
        if (self.testIlluminationSameAsReference.isChecked()):
            self.testIlluminationCondition.setValue(self.referenceIlluminationCondition.value())
            self.testIlluminationCondition.setDisabled(True)
        else:
            self.testIlluminationCondition.setDisabled(False)

        if (self.testEvaluationOptionsSameAsReference.isChecked()):
            self.useLuminanceOnlyTest.setChecked(self.useLuminanceOnlyReference.isChecked())
            self.applyReflectionTest.setChecked(self.applyReflectionReference.isChecked())
            self.applyScreenDimingTest.setChecked(self.applyScreenDimingReference.isChecked())
            self.useLuminanceOnlyTest.setDisabled(True)
            self.applyReflectionTest.setDisabled(True)
            self.applyScreenDimingTest.setDisabled(True)
        else:
            self.useLuminanceOnlyTest.setDisabled(False)
            self.applyReflectionTest.setDisabled(False)
            self.applyScreenDimingTest.setDisabled(False)

        self.simParamsTest = new_simul_params(illuminant=self.testIlluminationCondition.value(),
                                              use_luminance_only=self.useLuminanceOnlyTest.isChecked(),
                                              apply_reflection=self.applyReflectionTest.isChecked(),
                                              apply_screen_dimming=self.applyScreenDimingTest.isChecked())

    def updateTestDisplay(self):
        if (self.testDisplaySameAsReference.isChecked()):
            self.testDisplayLmax.setValue(self.referenceDisplayLmax.value())
            self.testDisplayContrast.setValue(self.referenceDisplayContrast.value())
            self.testDisplayLmax.setDisabled(True)
            self.testDisplayContrast.setDisabled(True)
        else:
            self.testDisplayLmax.setDisabled(False)
            self.testDisplayContrast.setDisabled(False)

        self.ddmTest = DisplayModel(L_max=self.testDisplayLmax.value(),
                                    contrast_ratio=self.testDisplayContrast.value())

    def updateAllDisplay(self):
        self.updateReferenceDisplay()
        self.updateTestDisplay()

    def QImageToCvMat(self, incomingImage):
        # '''  Converts a QImage into an opencv MAT format  '''
        incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGB32)
        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.constBits()
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
        return arr

    def simulateReferenceImage(self):
        self.simulatedReferenceImageL = self.ddmRef.display_simulation(
            rgb_view(self.imageReference).astype(np.float) / 255.,
            self.simParamsReference.illuminant,
            use_luminance_only=self.simParamsReference.use_luminance_only,
            inject_reflection=self.simParamsReference.apply_reflection,
            use_display_dimming=self.simParamsReference.apply_screen_dimming)

        self.simulatedReferenceImageView.display(array2qimage(self.simulatedReferenceImageL))

    def simulateTestImage(self):
        self.simulatedTestImageL = self.ddmTest.display_simulation(
            rgb_view(self.imageTest).astype(np.float) / 255,
            self.simParamsTest.illuminant,
            use_luminance_only=self.simParamsTest.use_luminance_only,
            inject_reflection=self.simParamsTest.apply_reflection,
            use_display_dimming=self.simParamsTest.apply_screen_dimming)

        self.simulatedTestImageView.display(array2qimage(self.simulatedTestImageL))

    def createSimulationResultsGroupBox(self):
        self.simulationResultsGroupBox = QGroupBox("Simulation Results")

        self.simulationResultsGroupBoxLayout = QHBoxLayout()

        self.createSimulatedReferenceGroupBox()
        self.createSimulatedTestGroupBox()
        self.createResultsMetricsGroupBox()

        self.simulationResultsGroupBoxLayout.addWidget(self.simulatedReferenceGroupBox)
        self.simulationResultsGroupBoxLayout.addWidget(self.simulatedTestGroupBox)
        self.simulationResultsGroupBoxLayout.addWidget(self.resultsMetricsGroupBox)
        self.evaluationMetricsGroupBoxLayout.addStretch(1)

        self.simulationResultsGroupBox.setLayout(self.simulationResultsGroupBoxLayout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
