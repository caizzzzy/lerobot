# Diana Robot Python SDK

A Python SDK for controlling Diana robotic arms, including C/C++ dynamic libraries and high-level Python interfaces.

## Features

- Easy-to-use Python API for Diana robots
- Supports joint and Cartesian motion commands
- Includes all required `.so` libraries for Linux
- Automatic environment setup for dynamic library loading

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/chongchongjjj/DianaRobotSDK.git
   cd DianaRobotSDK
   ```

2. **Install the package and set up environment variables:**
   ```bash
   chmod +x install_diana_robot.sh
   ./install_diana_robot.sh
   ```

   This script will:
   - Install the Python package to your current environment
   - Copy all `.so` files to the package directory
   - Add the required `LD_LIBRARY_PATH` to your `~/.bashrc`

3. **Restart your terminal or run:**
   ```bash
   source ~/.bashrc
   ```

## Usage Example

```python
from diana_robot.DianaRobot import DianaRobot

robot = DianaRobot(ip='192.168.1.10')
print("Current joints:", robot.getjoints())
robot.movej([0, -0.5, 0.5, 0, 0, 0, 0])
robot.movel([0.3, 0.1, 0.2, 0, 3.14, 0])
robot.stop()
robot.close()
```

## Directory Structure

```
diana_robot/
    __init__.py
    DianaApi.py
    DianaRobot.py
    libDianaApi.so
    libGenericAlgorithm.so
    libToolSdk.so
    libBasicSdk.so
    libVersionApi.so
    libxml2.so
setup.py
install_diana_robot.sh
README.md
.gitignore
```

## Requirements

- Python 3.8+
- Linux (x86_64)
- Diana robot hardware and network access

## Notes

- All `.so` files are required for the SDK to work.  
- If you change your Python environment, rerun `install_diana_robot.sh`.
- For advanced usage, see the API documentation in `DianaApi.py` and `DianaRobot.py`.

## License

Apache License 2.0

---

For issues or contributions, please open an issue or pull request on GitHub.