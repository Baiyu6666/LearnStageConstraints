# BarClean Real Robot Commands — iiwa7

This guide assumes an iiwa7 and the `iiwa7_new` site. Run the steps in order on the new Linux computer. Docker, Compose v2, Buildx, and conda must be available. The Docker build installs ROS and robot-side Python dependencies; host ROS and PyTorch are not required. The directory and Docker container names still contain `iiwa14`; the selected profile determines the actual robot model.

Commands can be copied after replacing the project path. Values marked as site-specific must be measured or confirmed locally. The iiwa7 site currently contains simulation examples, not a calibrated setup for your new robot.

## 1. Prepare the Host Python Environment

Replace the path and keep using the same terminal:

```bash
export BARCLEAN_ROOT="/absolute/path/to/BarCleanRuntime"
export ROBOT_PROFILE=iiwa7
export SITE_PROFILE=iiwa7_new
cd "$BARCLEAN_ROOT"
```

Create the environment only if it does not exist, then install dependencies:

```bash
conda create -n segment python=3.11 -y
conda activate segment
python -m pip install -r requirements.txt
```

The host environment runs the GUI, data processing, and learning. Docker uses its own Python environment. In a new terminal, repeat the exports and `conda activate segment`.

## 2. Configure the Robot and Site

```bash
cd "$BARCLEAN_ROOT/robot/stage_cons_iiwa14"
test -f .env || cp .env.example .env
ip -br link
nano .env
nano config/runtime/sites/iiwa7_new/site.json
```

Set these values in `.env`:

```dotenv
ROBOT_PROFILE=iiwa7
SITE_PROFILE=iiwa7_new
ROBOT_IFACE=REPLACE_WITH_ROBOT_NETWORK_INTERFACE
SCENE_POSE_SOURCE=fixed
```

In `site.json`, fill `network.host_ip`, `network.robot_ip`, and `network.host_prefix` using the cabinet's actual network configuration. The computer and robot need distinct usable IPs in the same subnet. These addresses are not detected automatically. `.env.source-machine` describes the old computer and robot; do not copy its values blindly.

Where to get these values:

- `ROBOT_IFACE`: run `ip -br link` before and after plugging in the cabinet Ethernet cable. With the cabinet powered on, identify the Ethernet interface whose link comes up, such as `enp3s0` or `enx...`. Use that name, not Wi-Fi or `docker0`.
- `robot_ip`: read the controller's IP for the connected Ethernet port in the cabinet/SmartPAD network configuration, or ask the lab operator. Do not use the old robot's IP without checking.
- `host_ip`: choose an unused computer address in the controller's subnet, and make it match the workstation destination IP configured in the cabinet's FRI application. To inspect the interface's current address, run `ip -br addr show dev enp3s0`, replacing `enp3s0` with your interface. If no address is assigned, `connect_robot_network.sh` will add the address you entered in `site.json`.
- `host_prefix`: convert the cabinet's subnet mask to its prefix length; `255.255.255.0` means `24`, and `255.255.0.0` means `16`. Use the actual mask, not an assumed default.

For live tracking, set `SCENE_POSE_SOURCE=optitrack` and update the site's `optitrack` server IP and rigid-body names. `fixed` uses saved scene poses and still requires correct calibration.

Before enabling hardware, configure and review:

- `config/runtime/sites/iiwa7_new/demo_scene.json` and `scenes/`: robot-frame table, bar and obstacle geometry, tracking transform, and initial IK seed.
- `config/runtime/sites/iiwa7_new/tasks/`: task poses and Home for this installation.
- `config/runtime/sites/iiwa7_new/assistance.yaml`: demonstration assistance settings.
- `config/runtime/robots/iiwa7.json`: velocity, acceleration, and torque limits.
- The controller's compatible FRI application, workstation destination IP, tool and payload settings, and robot readiness. Match the application's port and update period to the driver configuration.

`hardware_commissioned` is already true at your request, so there is no additional manual release step. This flag does not perform calibration or validate the physical installation; the network and on-site checks above still apply.

## 3. Build and Open the GUI

The Docker build requires an SSH key authorized to access `epfl-lasa/kuka_fri`:

```bash
ssh-add -l
git ls-remote git@github.com:epfl-lasa/kuka_fri.git HEAD
```

If an agent/key is missing, add your authorized key, replacing the path if necessary, then retry the repository check:

```bash
eval "$(ssh-agent -s)"
ssh-add "$HOME/.ssh/id_ed25519"
git ls-remote git@github.com:epfl-lasa/kuka_fri.git HEAD
```

Build the shared image, start the base container, and install the GUI service:

```bash
cd "$BARCLEAN_ROOT/robot/stage_cons_iiwa14"
./scripts/start.sh
STAGE_HOST_PYTHON="$(command -v python)" ./scripts/install_host_gui_service.sh
```

Open <http://127.0.0.1:8080> and verify **Robot: iiwa7 / Site: iiwa7_new**. `start.sh` automatically builds the image and creates the container; it does not start a robot driver. The GUI manages demonstration and execution stations.

If reinstalling from a new project path while the service is already running, restart it:

```bash
systemctl --user restart stage-cons-supervisor.service
```

Build or restart only with FRI inactive and no motion, recording, or task in progress. Profile/configuration changes require restarting the workstation and GUI before use.

## 4. Connect and Collect Demonstrations

With the dedicated robot Ethernet cable connected and site IPs configured:

```bash
cd "$BARCLEAN_ROOT/robot/stage_cons_iiwa14"
./scripts/connect_robot_network.sh
```

The script configures the selected host interface and pings the robot; it does not start FRI. Before hardware operation, complete section 2, confirm the robot is mastered and fault-free, and keep the E-stop available.

In the GUI:

1. Select `Bar Clean`, Scene, and Scene pose source.
2. Click `Enter Demo`; follow the SmartPAD instructions for `TORQUE` mode.
3. Wait for readiness, enter the label, then click `Start Recording`.
4. Guide the robot through the demonstration.
5. Click `Stop and Save Rosbag`, then `Exit Demo`.

Recordings are saved under:

```text
robot/stage_cons_iiwa14/data/demos/iiwa7/iiwa7_new/BarClean/<timestamp_label>/
    demo.bag
    metadata.json
```

Leave `Record video` unchecked unless the lab camera is configured.

## 5. Process Demonstrations and Learn

Replace `NEW_RECORDING_DIR` with the recorded directory name. The dataset directory below matches this site's initial learning configuration:

```bash
cd "$BARCLEAN_ROOT"
conda activate segment
python experiments/bar_clean_data_pipeline.py run \
  robot/stage_cons_iiwa14/data/demos/iiwa7/iiwa7_new/BarClean/NEW_RECORDING_DIR/demo.bag \
  --dataset-dir robot/stage_cons_iiwa14/data/processed/iiwa7/iiwa7_new
```

Review `Start`, `CP1–CP4`, and `End`, click `Confirm all`, select `Activate`, then click `Export`. This exports `training_5hz.npz` and updates `config/runtime/sites/iiwa7_new/learning.json`. To reopen the review:

```bash
python experiments/bar_clean_data_pipeline.py review \
  robot/stage_cons_iiwa14/data/processed/iiwa7/iiwa7_new
```

Use a desktop graphical session. If Tk is missing, install it with `conda install -n segment tk -y`. Do not use `--overwrite` unless discarding the existing dataset review is intended.

Train with one method seed:

```bash
python runners/run_benchmark.py \
  --methods map_balanced_pooled \
  --datasets BarClean \
  --method-seeds 0 \
  --outdir outputs/benchmark/iiwa7/iiwa7_new/barclean_real_map_balanced_pooled
```

The model is saved at:

```text
outputs/iiwa7/iiwa7_new/map_balanced_pooled/BarClean/method_seed_000/learned_constraints.json
```

The GUI discovers planning models in `outputs`. For a MAP/SWCL comparison, use `--methods map,swcl` with the same single seed; SWCL results are not directly selectable planning models. Add `--resume` to reuse completed results when the dataset fingerprint matches.

The included dataset and legacy model were collected on the old iiwa14 installation. They are not new iiwa7 demonstrations. The iiwa7 learning profile initially points to a dataset that must be collected and exported. Skip collection/training only if you already have a model whose task geometry and feature definitions have been checked for this site.

## 6. Simulate, Then Execute

In the GUI:

1. Select `Bar Clean` and a checked MAP model or `True constraints`.
2. Verify scene, Start, and Goal in the iiwa7 base frame. Positions use meters; Roll/Pitch/Yaw use degrees.
3. Click `Simulate` and inspect the trajectory and execution status.
4. End Demo mode and release control.
5. Follow the SmartPAD instructions for `POSITION` mode, click `Execute`, and complete the hardware confirmation.
6. Use `Stop Task` to abort when needed.

`True constraints` also uses the site's task configuration, which must be calibrated. Simulation can run while `hardware_commissioned` is false. It is not hardware validation and does not model complete debris-sweeping physics. Do not start a second FRI driver or bypass the GUI's execution checks. Use the GUI's `Output` field to locate saved task data.

## 7. Finish or Check Logs

Stop and save recording, stop the task, release robot control, and confirm the robot has stopped. Then stop the workstation through the GUI or terminal:

```bash
cd "$BARCLEAN_ROOT/robot/stage_cons_iiwa14"
export USER_UID="$(id -u)"
./scripts/stop.sh
```

Saved data and the Docker image are retained. For errors:

```bash
journalctl --user -u stage-cons-supervisor.service -n 100 --no-pager
docker logs --tail 100 stage_cons_iiwa14
```

Fix the first reported error before retrying. GUI readiness indicators show missing robot, FRI, tracking, or recorder inputs.
