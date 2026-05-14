#helps in building ml as package itself

from setuptools import find_packages, setup
from typing import List
HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->list[str]:
    #this bfunction will return the list of requirements
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements if req.strip() and not req.startswith("#")]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
setup(
    name='ml_project',
    version='0.1',
    author="Farooque",
    author_email="2306033@kiit.ac.in",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )