from setuptools import setup, find_packages

setup(name="qelos-util",
      description="qelos-util",
      author="Sum-Ting Wong",
      author_email="sumting@wo.ng",
      install_requires=["scipy",
                        "unidecode",
                        "tqdm",
                        "ujson",
                        "matplotlib",
                        "fire",
                        "optuna"
                        ],
      packages=["qelos"],
      )
