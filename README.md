# Soft Actor Critic (SAC)
## Overview 

This repository contains an implementation of Soft Actor-Critic (SAC), a state-of-the-art deep reinforcement learning algorithm designed for continuous control tasks. Originally proposed in the paper ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" by Haarnoja et al](https://arxiv.org/abs/1801.01290), SAC enhances traditional actor-critic methods by integrating ideas from maximum entropy reinforcement learning and stochastic policies.  

SAC leverages the advantages of entropy regularization to promote exploration while maintaining a balance between exploration and exploitation. This is achieved through the incorporation of an entropy term in the objective function, encouraging policies to be both high-performing and diverse. The codebase has been evaluated on a variety of standard continuous control environments available in Gymnasium and MuJoCo libraries.

## Setup

### Required Dependencies

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### Running the Algorithm

You can run the algorithm on any supported Gymnasium environment. For example:

```bash
python main.py --env 'LunarLanderContinuous-v2'
```

--- 

<table>
    <tr>
        <td>
            <p><b>Pendulum-v1</b></p>
            <img src="environments/Pendulum-v1.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>LunarLanderContinuous-v2</b></p>
            <img src="environments/LunarLanderContinuous-v2.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>MountainCarContinuous-v0</b></p>
            <img src="environments/MountainCarContinuous-v0.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/Pendulum-v1_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/LunarLanderContinuous-v2_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/MountainCarContinuous-v0_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>BipedalWalker-v3</b></p>
            <img src="environments/BipedalWalker-v3.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Hopper-v4</b></p>
            <img src="environments/Hopper-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Humanoid-v4</b></p>
            <img src="environments/Humanoid-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BipedalWalker-v3_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Hopper-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Humanoid-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Ant-v4</b></p>
            <img src="environments/Ant-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>HalfCheetah-v4</b></p>
            <img src="environments/HalfCheetah-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>HumanoidStandup-v4</b></p>
            <img src="environments/HumanoidStandup-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/Ant-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/HalfCheetah-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/HumanoidStandup-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>InvertedDoublePendulum-v4</b></p>
            <img src="environments/InvertedDoublePendulum-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>InvertedPendulum-v4</b></p>
            <img src="environments/InvertedPendulum-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Pusher-v4</b></p>
            <img src="environments/Pusher-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/InvertedDoublePendulum-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/InvertedPendulum-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Pusher-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Reacher-v4</b></p>
            <img src="environments/Reacher-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Swimmer-v3</b></p>
            <img src="environments/Swimmer-v3.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Walker2d-v4</b></p>
            <img src="environments/Walker2d-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/Reacher-v4_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Swimmer-v3_running_avg.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/Walker2d-v4_running_avg.png" width="250" height="250"/>
        </td>
    </tr>
</table>



## Acknowledgements

Special thanks to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).