from setuptools import setup, find_packages

setup(
    name="rl_path_planning",
    version="1.0.0",
    author="Duy-Nam Bui",
    author_email="duynam.robotics@gmail.com",
    description="A brief description of your project",
    url="https://github.com/duynamrcv/rl_path_planning",
    packages=find_packages(include=["*"]),  # Automatically discover all packages
    python_requires=">=3.6",  # Adjust to match your Python version support
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ],
)
