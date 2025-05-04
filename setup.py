from setuptools import setup, find_packages

setup(
    name="math_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytesseract>=0.3.10",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "google-generativeai>=0.3.0",
        "sympy>=1.12",
        "matplotlib>=3.7.0",
        "scikit-image>=0.21.0"
    ],
    python_requires=">=3.8",
) 