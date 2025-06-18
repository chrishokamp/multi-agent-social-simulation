from setuptools import setup, find_packages

setup(
    name="multi_agent_social_simulation",
    version="0.1.0",
    description="A multi-agent social simulation framework",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)