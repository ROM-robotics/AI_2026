"""Setup script for LLM Studio."""

from setuptools import setup, find_packages

setup(
    name="llm-studio",
    version="1.0.0",
    description="A Terminal-based LLM Management Studio",
    author="LLM Studio",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "textual>=0.47.0",
        "rich>=13.7.0",
        "llama-cpp-python>=0.2.50",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "huggingface-hub>=0.20.0",
        "pydantic>=2.5.0",
        "httpx>=0.26.0",
        "pyyaml>=6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "llm-studio=llm_studio.app:main",
        ],
    },
    package_data={
        "llm_studio": ["ui/styles/*.tcss"],
    },
)
