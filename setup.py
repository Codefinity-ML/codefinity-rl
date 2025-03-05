from setuptools import setup, find_packages

setup(
    name="codefinity-rl",
    version="0.1.0",
    description="RL environmets for course tasks on Codefinity platform",
    packages=find_packages(),
    install_requires=["numpy", "gymnasium", "jupyter", "pygments", "pillow", "pygame"],
)
