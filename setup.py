from setuptools import setup, find_packages

setup(
    # Basic info
    name='network_spatial_coherence',
    version='0.1.0',
    author='David Fernandez Bonet',
    author_email='dfb@kth.se',
    url='https://github.com/DavidFernandezBonet/Spatial Constant Analysis',
    description='Network Validation using the Spatial Coherence Framework.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This is important for markdown to render correctly

    # Package info
    packages=find_packages(exclude=('tests', 'docs')),  # Automatically find and include all packages
    install_requires=[
        'matplotlib==3.8.3',
        'memory_profiler==0.61.0',
        'networkx==3.2.1',
        'nodevectors==0.1.23',
        'numpy==1.26.4',
        'pandas==2.2.1',
        'pecanpy==2.0.9',
        'Pillow==10.2.0',  #
        'pixelator==1.2.0',
        'plotly==5.19.0',
        'pymde==0.1.18',
        'python_igraph==0.11.4',
        'scienceplots==2.1.1',
        'scikit_learn==1.3.1',
        'scipy==1.12.0',
        'seaborn==0.13.2',
        'torch==2.1.0',
        'umap-learn',
    ],

    # Optional package data
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        'your_package_name': ['*.txt', '*.rst'],
        # And include any files found in the 'data' subdirectory of the 'your_package_name' package, also:
        'your_package_name': ['data/*'],
    },

    # Could be useful to create a script accessible from the command line
    entry_points={
        'console_scripts': [
            'your_command=your_package.module:function',
        ],
    },

    # Additional metadata
    classifiers=[
        'Development Status :: 3 - Alpha',  # Change as appropriate
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',  # Change as appropriate
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Additional requirements
    python_requires='>=3.6',  # Minimum version requirement of the Python interpreter
    extras_require={
        'dev': [
            'pytest>=3.7',
            'check-manifest',
            'twine',
            # etc.
        ],
    },
)
