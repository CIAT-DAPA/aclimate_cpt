from setuptools import setup, find_packages

setup(
    name="aclimate_cpt",
    version='v0.0.15',
    author="stevensotelo",
    author_email="h.sotelo@cgiar.com",
    description="Prediction module",
    url="https://github.com/CIAT-DAPA/aclimate_cpt",
    download_url="https://github.com/CIAT-DAPA/aclimate_cpt",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    keywords='prediction aclimate',
    entry_points={
        'console_scripts': [
            'aclimate_cpt=aclimate_cpt.aclimate_run_cpt:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas==1.4.2",
        "tqdm==4.62.3",
        "numpy==1.25.2",
        "opencv-python==4.8.0.76",
        "matplotlib==3.5.1",
        "requests==2.27.1",
        "datetime==4.3",
        "xarray==0.20.2",
        "dask==2023.1.0",
        "monthdelta==0.9.1"
    ]
)