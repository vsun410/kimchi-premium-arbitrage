"""Setup configuration for Kimchi Premium Arbitrage System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="kimchi-premium-arbitrage",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Kimchi Premium Futures Hedge Arbitrage System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kimchi-premium-arbitrage",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "ccxt[ws]>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "cryptography>=41.0.0",
        "aiohttp>=3.8.0",
        "prometheus-client>=0.17.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "xgboost>=1.7.0",
            "scikit-learn>=1.3.0",
            "stable-baselines3>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kimchi-monitor=scripts.monitor_kimchi_premium:main",
            "check-rate=scripts.check_current_rate:main",
        ],
    },
)