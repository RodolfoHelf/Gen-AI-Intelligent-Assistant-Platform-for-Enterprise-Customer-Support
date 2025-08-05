#!/usr/bin/env python3
"""
Setup script for Gen-AI Assistant Platform
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gen-ai-assistant",
    version="1.0.0",
    author="Gen-AI Assistant Team",
    author_email="support@genai-assistant.com",
    description="An intelligent customer service platform powered by Large Language Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/gen-ai-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "dashboard": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
        ],
        "ml": [
            "tensorflow>=2.15.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gen-ai-assistant=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="ai, machine-learning, customer-support, llm, openai, langchain",
    project_urls={
        "Bug Reports": "https://github.com/your-org/gen-ai-assistant/issues",
        "Source": "https://github.com/your-org/gen-ai-assistant",
        "Documentation": "https://github.com/your-org/gen-ai-assistant/wiki",
    },
) 