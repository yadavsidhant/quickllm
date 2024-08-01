from setuptools import setup, find_packages

setup(
    name="quickllm",
    version="0.1.6",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.0.0",
        "torch>=1.0.0",
        "pandas>=1.0.0",
        "matplotlib>=3.0.0",
        "seaborn>=0.11.0",
        "datasets>=1.0.0",
        "colorama",
    ],
    author="Sidhant Yadav",
    author_email="supersidhant10@gmail.com",
    description="A library for quick fine-tuning and interaction with popular language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yadavsidhant/quickllm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    project_urls={
        'Bug Tracker': 'https://github.com/yadavsidhant/quickllm/issues',
        'Documentation': 'https://github.com/yadavsidhant/quickllm#readme',
        'Youtube': 'https://www.youtube.com/@SidhantKYadav',
    },
)
