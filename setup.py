from setuptools import setup, find_packages

setup(
    name='attention_head_intervention',                # You can name this anything
    version='0.1',                    # Your project version
    packages=find_packages(where='src'),
    package_dir={'': 'src'},         # Tell Python your code is in src/
    install_requires=[],             # Add dependencies here, if any
)