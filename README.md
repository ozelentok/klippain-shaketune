# Klipper ShakeTune Module

Proof of concept for turning [Klippain "Shake&Tune"](https://github.com/Frix-x/klippain-shaketune) into a fully integrated Klipper module.

ShakeTune is a module designed to automate and calibrate the input shaper system on your Klipper 3D printer with a streamlined workflow and insightful vizualisations.

It operates in two steps:
  1. Utilizing specially tailored Klipper macros, it initiates tests on either the belts or the printer X/Y axis to measure the machine axes behavior. This is basically an automated call to the Klipper `TEST_RESONANCES` macro with custom parameters.
  2. Then a custom Python script is called to:
     1. Generate insightful and improved graphs, aiding in parameter tuning for the Klipper `[input_shaper]` system (including best shaper choice, resonant frequency and damping ratio) or diagnosing and rectifying mechanical issues (like belt tension, defective bearings, etc..)
     2. Relocates the graphs and associated CSV files to your Klipper config folder for easy access via Mainsail/Fluidd to eliminate the need for SSH.
     3. Manages the folder by retaining only the most recent results (default setting of keeping the latest three sets).

Check out the **[detailed documentation of the ShakeTune module here](./docs/README.md)**. You can also look at the documentation for each type of graph by directly clicking on them below to better understand your results and tune your machine!

| [Belts graph](./docs/macros/belts_tuning.md) | [Axis input shaper graphs](./docs/macros/axis_tuning.md) | [Vibrations graph](./docs/macros/vibrations_tuning.md) |
|:----------------:|:------------:|:---------------------:|
| [<img src="./docs/images/belts_example.png">](./docs/macros/belts_tuning.md) | [<img src="./docs/images/axis_example.png">](./docs/macros/axis_tuning.md) | [<img src="./docs/images/vibrations_example.png">](./docs/macros/vibrations_tuning.md) |

  > The original Klippain-Shake&Tune module used [Gcode shell command plugin](https://github.com/dw-0/kiauh/blob/master/docs/gcode_shell_command.md) to execute commands which has a great potential for abuse and hacks via remote code execution.
  > This version solves that issue and executes everything inside the main Klipper process and does not require the shell command plugin.

## Installation

Follow these steps to install the ShakeTune module in your printer:
  1. Be sure to have a working accelerometer on your machine. You can follow the official [Measuring Resonances Klipper documentation](https://www.klipper3d.org/Measuring_Resonances.html) to configure one. Validate with an `ACCELEROMETER_QUERY` command that everything works correctly.
  1. Clone this repository on your klipper machine
  1. Run the install script with Klipper's installation directory and config directory as arguments
     ```bash
      # Replace /usr/lib/klipper with your klipper installion directory
     ./install.sh /usr/lib/klipper /etc/klipper
     ```
  1. Append the following to your `printer.cfg` file and restart Klipper:
     ```
     [shaketune]
     [include shaketune/*.cfg]
     ```
  - The installer patches klipper code so `ACCELEROMETER_MEASURE` will wait for output files to finish writing to avoid incomplete file reading.
    - The original ShakeTune code iterated through many /proc/*/fds to check if the CSV file is still opened by any process, that was because Klipper would spawn a new process just to write the CSV, meaning it can't just check its own process file descriptors (To avoid heavy CPU usage iterating all those virtual files, the original code checked every 2 seconds).
  , )

## Usage
Ensure your machine is homed, then invoke one of the following macros as needed:
  - `SHAKETUNE_MAP_AXES` to automatically find Klipper's `axes_map` parameter for your accelerometer orientation
  - `SHAKETUNE_TEST_BELTS` for belt resonance graphs, useful for verifying belt tension and differential belt paths behavior.
  - `SHAKETUNE_TEST_RESONANCES` for input shaper graphs to mitigate ringing/ghosting by tuning Klipper's input shaper system.
  - `SHAKETUNE_TEST_VIBRATIONS` for machine and motors vibration graphs, used to optimize your slicer speed profiles and TMC drivers parameters.
  - `SHAKETUNE_SHAKE_AXIS` to sustain a specific excitation frequency, useful to let you inspect and find out what is resonating.

For further insights on the usage of these macros and the generated graphs, refer to the [ShakeTune module documentation](./docs/README.md).
