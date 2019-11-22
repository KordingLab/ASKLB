from setuptools import setup

setup(
    name='asklb',
    packages=['asklb'],
    include_package_data=True,
    # TODO update this with our dev requirements
    install_requires=[
        'flask',
    ],
)