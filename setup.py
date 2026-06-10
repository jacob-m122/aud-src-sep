from setuptools import setup, find_packages

setup(
    name='audio_separation_project',
    version='0.1.0',
    description='Audio Stem Source Separation via Cross-Track Attention',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchaudio',
        'torchmetrics',
        'librosa',
        'matplotlib',
        'soundfile',
        'scipy',
        'numpy',
        'jupyter',
        'kaggle'
    ],
)