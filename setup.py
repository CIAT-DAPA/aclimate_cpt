from setuptools import setup, find_packages

setup(
    name="aclimate_prediction",
    version='v1.0.0',
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
            'prediction=aclimate_prediction.aclimate_run_cpt:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy==1.22.3",
        "pandas==1.4.2",
        "matplotlib==3.5.1",
        "requests==2.27.1",
        "shutil==3.4.3",
        "tqdm==4.62.3",
        "concurrent.futures==1.10.0",
        "itertools==7.10.0",
        "monthdelta==2.1.0",
        "datetime==4.3",
        "json==2.2.1",
        "subprocess==0.7.0",
        "tempfile==0.8.1",
        "csv==1.55",
        "sys==3.10.0",
        "date==3.0.0",
        "dask==2023.0.2",
        "xarray==0.20.2"
    ]
)