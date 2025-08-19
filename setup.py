from setuptools import setup, find_packages

setup(
    name="codefinity-rl",
    version="0.2.1",
    description="RL environmets for course tasks on Codefinity platform",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gymnasium",
        "jupyter",
        "pygments",
        "pillow",
        "pygame",
        "matplotlib",
        "seaborn",
        "imageio",
    ],
)
