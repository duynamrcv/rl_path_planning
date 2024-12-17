from setuptools import setup, find_packages

setup(
    name="my_project",  # Replace with your project's name
    version="1.0.0",  # Replace with your project's version
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/my_project",  # Replace with your project's repository URL
    packages=find_packages(include=["*"]),  # Automatically discover all packages
    install_requires=[
        # List your project's dependencies here, e.g.:
        # "numpy>=1.20.0",
    ],
    python_requires=">=3.6",  # Adjust to match your Python version support
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # Includes files from MANIFEST.in if needed
    zip_safe=False,
)
