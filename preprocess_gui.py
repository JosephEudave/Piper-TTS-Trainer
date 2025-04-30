import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                            QFileDialog, QMessageBox, QProgressBar, QCheckBox,
                            QGroupBox, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from preprocess_audio import process_directory, validate_metadata

class Worker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, input_dir, output_dir, metadata_path, target_sr, single_speaker):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.metadata_path = metadata_path
        self.target_sr = target_sr
        self.single_speaker = single_speaker

    def run(self):
        try:
            # Validate metadata first
            self.progress.emit("Validating metadata...")
            is_valid, errors = validate_metadata(self.metadata_path, self.input_dir, self.single_speaker)
            if not is_valid:
                self.error.emit("\n".join(errors))
                return

            # Process audio files
            self.progress.emit("Processing audio files...")
            results = process_directory(self.input_dir, self.output_dir, self.target_sr)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piper TTS Audio Preprocessor")
        self.setMinimumWidth(600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Input directory
        input_group = QGroupBox("Input Directory")
        input_layout = QHBoxLayout()
        self.input_dir = QLineEdit()
        self.input_dir.setReadOnly(True)
        input_btn = QPushButton("Browse")
        input_btn.clicked.connect(lambda: self.browse_directory(self.input_dir))
        input_layout.addWidget(self.input_dir)
        input_layout.addWidget(input_btn)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setReadOnly(True)
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(lambda: self.browse_directory(self.output_dir, True))
        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(output_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Metadata file
        metadata_group = QGroupBox("Metadata File")
        metadata_layout = QHBoxLayout()
        self.metadata_file = QLineEdit()
        self.metadata_file.setReadOnly(True)
        metadata_btn = QPushButton("Browse")
        metadata_btn.clicked.connect(self.browse_metadata)
        metadata_layout.addWidget(self.metadata_file)
        metadata_layout.addWidget(metadata_btn)
        metadata_group.setLayout(metadata_layout)
        layout.addWidget(metadata_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        # Sample rate
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("Sample Rate:"))
        self.sr_22050 = QRadioButton("22050 Hz")
        self.sr_16000 = QRadioButton("16000 Hz")
        self.sr_22050.setChecked(True)
        sr_layout.addWidget(self.sr_22050)
        sr_layout.addWidget(self.sr_16000)
        options_layout.addLayout(sr_layout)
        
        # Speaker type
        self.single_speaker = QCheckBox("Single Speaker Dataset")
        self.single_speaker.setChecked(True)
        options_layout.addWidget(self.single_speaker)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Status label
        self.status = QLabel()
        layout.addWidget(self.status)
        
        # Process button
        self.process_btn = QPushButton("Process Audio Files")
        self.process_btn.clicked.connect(self.process_files)
        layout.addWidget(self.process_btn)
        
        # Results text area
        self.results = QLabel()
        self.results.setWordWrap(True)
        layout.addWidget(self.results)

    def browse_directory(self, line_edit, is_output=False):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            line_edit.setText(dir_path)
            if is_output and not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def browse_metadata(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Metadata File", "", "CSV Files (*.csv);;Text Files (*.txt)"
        )
        if file_path:
            self.metadata_file.setText(file_path)

    def process_files(self):
        # Validate inputs
        if not all([self.input_dir.text(), self.output_dir.text(), self.metadata_file.text()]):
            QMessageBox.critical(self, "Error", "Please fill in all fields")
            return

        # Get sample rate
        target_sr = 22050 if self.sr_22050.isChecked() else 16000

        # Create worker thread
        self.worker = Worker(
            self.input_dir.text(),
            self.output_dir.text(),
            self.metadata_file.text(),
            target_sr,
            self.single_speaker.isChecked()
        )
        
        # Connect signals
        self.worker.progress.connect(self.update_status)
        self.worker.finished.connect(self.process_finished)
        self.worker.error.connect(self.show_error)
        
        # Disable process button and show progress
        self.process_btn.setEnabled(False)
        self.progress.setVisible(True)
        
        # Start worker
        self.worker.start()

    def update_status(self, message):
        self.status.setText(message)

    def process_finished(self, results):
        self.process_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        # Show results
        result_text = "Processing complete!\n\nResults:\n"
        for file, msg in results:
            result_text += f"{file}: {msg}\n"
        self.results.setText(result_text)
        
        QMessageBox.information(self, "Success", "Processing completed successfully!")

    def show_error(self, error_message):
        self.process_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", error_message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 