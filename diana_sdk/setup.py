from setuptools import setup, find_packages

setup(
    name='diana_robot',
    version='1.0.0',
    description='Diana Robot Python SDK',
    author='Zilong Huang',
    packages=find_packages(),
    package_data={
        'diana_robot': [
            'libDianaApi.so',
            'libGenericAlgorithm.so',
            'libToolSdk.so',
            'libBasicSdk.so',
            'libVersionApi.so',
            'libxml2.so'
        ]
    },
    install_requires=[],
    include_package_data=True,
)