# Readme of Supplementary Software File

This is the code repository for article "Human-Guided Continual Learning for Multifaceted Improvement of Self-driving Vehicles"

## 1. Structure

The files are structured as follows, where the note in the brackets briefly explains the content:
```
+-- Statistics [Codes and Data in the article for Visualization]
|   +-- ethic_risk
|   |   +-- ethic_risk_calc.py [Calculate ethical risk]
|   +-- source_data
|   |   +-- ethical_cases [Run the results in Test1]
|   |   +-- corner_cases [Run the results in Test2]
|   |   +-- routine_cases [Run the results in Test3]
|   |   +-- variable_cases [Run the results in Test4]
+-- Framework [Codes in the article for the simulations]
|   +-- DMS
|   |   +-- DMS.py [Run for the DMS]
|   +-- traj_analysis
|   |   +-- traj_analysis.py [Run for the human guidance analysis]
|   +-- HugCL
|   |   +-- inference.py [Run for the HugCL agent in simulations]
|   |   +-- scenario_runner.py [Run for the simulation environment]
+-- Deployment [Codes in the article for the field testing]
|   +-- HDmap
|   |   +-- map_parser.py [Run for the map processing]
|   +-- cl_inference
|   |   +-- inference_autoware_wrapper.py [Run for the HugCL agent in the field testing]
```

## 2. Render Test Results

Most raw data are readily visualizable using Office Excel's native tools.

Remaining raw data can be visualized by running the provided source code (.m), written in [Matlab](https://www.mathworks.com/products/matlab.html)>=R2023b.

## 3. Light DEMO Codes for Simulations

This DEMO shows the process of running simulation tests.
The function is to generate the interactive environment and decide the final trajectory.

### 3.1 Installation

The package is written in Python 3. The usage of the [Anaconda](https://www.anaconda.com/distribution/#download) Python distribution is recommended.

After you have installed Python 3.8.1 or above, open a terminal in the Supplementary Software File and execute the following command to install the software (including the dependencies):

- Set Up Python Environment
    ```bash
    pip install -r ./requirements.txt
    ```

- Download [CARLA] (https://github.com/carla-simulator/carla/releases)>=0.9.13

- Run CARLA in the downloaded CARLA folder

    ```bash
    .\CarlaUE4.exe
    ```

### 3.2 Usage

- Run the interactive environment:

    ```bash
    python .\HugCL\scenario_runner\scenario_runner.py
    ```

- Then, open another terminal and run the HugCL agent:

    ```bash
    python .\HugCL\inference.py
    ```

### 3.3 More Data for evaluation

You can collect more data for the evaluation. 

#### 3.3.1 CARLA-based Data Collection

- Run the interactive environment:

    ```bash
    python .\HugCL\scenario_runner\scenario_runner.py
    ```

- Then, open another terminal and run the HugCL agent:

    ```bash
    python .\HugCL\inference.py --data_collection
    ```

Continuous human trajectories collected from the steering wheel and pedals need to be parsed into discrete decisions.

-  Run the trajectory analysis:

     ```bash
     python .\traj_analysis\traj_analysis.py
     ```
We also provide a keyboard-based approach for directly collecting high-level decisions.

#### 3.3.2 Instruction for other data:

A single data point is formulated into {state, action, cumulative rewards, done}

State contains ego information, surrounding users information, traffic regulations and the destination.

After that, when given a state, the HugCL agent can generate an action according to the collected data. 


## 4. Codes for Field Testing

### 4.1 Dependencies

- [ubuntu](https://releases.ubuntu.com/22.04/)==22.04 and x86 platform.
- [docker](https://www.docker.com/)>=20.10.16
- [autoware](https://github.com/autowarefoundation/autoware_universe/tree/humble)==universe
- [ROS2](https://docs.ros.org/en/humble/Installation.html)==humble
- [rviz2](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html)

### 4.2 Installation

#### 4.2.1 Install ros2:

Official tutotial can be found here: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

- setup sources
    ```bash
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    ```

- setup keys

    ```bash
    sudo apt update && sudo apt install curl -y
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" 
    sudo dpkg -i /tmp/ros2-apt-source.deb
    ```

- update apt source

    ```bash
    sudo apt update
    ```

- install ros-humble-desktop-full

    ```bash
    sudo apt install ros-humble-desktop
    ```

- activate ros environment

    ```bash
    source /opt/ros/humble/setup.bash
    ```

- rviz2 is default installed.

- This install takes about 20 minutes.

#### 4.2.2 Install docker:

- Docker is default on ubuntu. If docker is not available, following this official tutorial to install docker on Linux: https://docs.docker.com/engine/install/ubuntu/.
  
- This install takes about 10 minutes.

#### 4.2.3 Install autoware

Official tutotial can be found here: https://autowarefoundation.github.io/autoware-documentation/

- Download autowarefoundation/autoware in the AUTOWARE folder
    ```bash
    git clone https://github.com/autowarefoundation/autoware.git
    ```

- Open the another terminal, and build docker images locally
     ```bash
    ./docker/build.sh --devel-only
    ```

- This install takes about 40 minutes.

We also provided a pre‑built version for docker image, which can be downloaded from Google Cloud.

- Build autoware workspace
     ```bash
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    ```

- This install takes about 30 minutes.

#### 4.2.5 Download field data records

- Download the field testing rosbag data from Google Cloud.
  
- Unzip the file and copy the rosbags to desired location.

    ```bash
    unzip HugCL_data.zip
    cd data
    cp -r rosbag/ PATH_TO_PROJECT/Supplementary_Software/Deployment/cl_inference/
    ```

### 4.3 Parse the high-definition map

We provide a miniminal  high-definition map and its parsing algorithm.

-  Run the map parser:

     ```bash
     python .\Deployment\HDmap\map_parser.py
     ```

### 4.4 Launch the field testing

- Run autoware in the downloaded AUTOWARE folder
   
    ```bash
    source .\opt\ros\humble\setup.bash
    source .\install\setup.bash
    ```

- Trap termination signals to gracefully shutdown background processes
   
    ```bash
    trap 'echo "Stopping..."; kill $AUTOWARE_PID; exit' SIGINT SIGTERM
    ```

- Launch autoware in the background
   
    ```bash
    ros2 launch autoware_launch autoware.launch.xml \
    ```

- Open another terminal and run the HugCL agent inference script
    ```bash
    python3 inference_autoware_wrapper.py
    ```
