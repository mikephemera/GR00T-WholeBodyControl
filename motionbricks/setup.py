from setuptools import setup, find_packages

setup(
    name="motionbricks",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
        # MuJoCo 3.11's Linux GLFW teardown can segfault after the viewer closes.
        # 3.3.7 is verified with both headless and X11 MotionBricks demos.
        "mujoco==3.3.7",
        "scipy",
        "hydra-core",
        "omegaconf",
        "pytorch-lightning",
        "transformers",
        "pynput",
        "matplotlib",
        "vector-quantize-pytorch",
        "colorlog",
        "adam-atan2-pytorch",
    ],
)
