# Instructions for setting up the project environment

## Download & install Anaconda
1. Download Anaconda Python from [anaconda.com](https://www.anaconda.com/download#downloads). The below remarks refer to the command line version of the installer.
1. Run the installer, but first read through **all** of the below remarks.
  - If on Windows, you might need to run the downloaded installation file as administrator, depending on choice of installation directory (right-click the file, and click "Run as administrator").
  - Do not use the default installation path, if it contains spaces or other special characters. We do not know exactly when this could turn out to be an issue, but it is discouraged by Anaconda (more info [here](https://docs.anaconda.com/anaconda/user-guide/faq/#distribution-faq-windows-folder)). For instance, if John Doe ran the installer, the default path might be `C:\Users\John Doe\Anaconda3`, which could be an issue since his username has a space in it. An alternative would be `C:\anaconda3`, although this would probably require you to run the installation as administrator.
  - During the installation, make sure to select the option to add Anaconda to your PATH environment variable. As far as we know, this is usually not recommended by Anaconda and thus deselected by default.
  - You do not need to "Register Anaconda as the default/system Python 3.9"
3. After installation, run the command `conda init` (or optionally `conda init <SHELL_OF_YOUR_CHOICE>`), in order to setup conda to work properly with your command-line interpreter (terminal). You may have been asked for the `conda init` step to be automatically carried out during installation, but if not it is essential that you do it manually, in order to be able to activate your Conda environment later.
1. Close all terminal windows currently open (the `conda init` step above will not have any effect on existing terminal windows - only on new ones).

## Install Git and clone project repository
1. Install Git, https://git-scm.com/book/en/v2/Getting-Started-Installing-Git.
1. Start up a terminal (On Windows, use Powershell)
1. Clone the project GitHub repo, by the following command (this creates a folder in your current directory with the necessary files for the next instructions): `git clone https://github.com/danilorar/gnn_trajectory.git`

**Note:** The notebooks locate the data through the path `~/Desktop/gnn_trajectory/data`. Either run the clone command above from your `Desktop` folder, or change the `DATA_DIR` variable at the top of each notebook to match where you cloned the repo.

## Keeping the repo in sync
Cloning a repo creates a local copy of the Github repo on your computer. Whenever new changes are uploaded, or *pushed*,
to the Github repo you can get the new changes by *pulling* them to your local version with the `git pull` command.
If you have made changes in the same files as the new version on Github, there could be a conflict and git would not be able to automatically *merge* the changes.
To avoid this, *commit* (or `git stash`) your local changes before pulling, and resolve any remaining conflicts by hand.

## Create a Conda environment and install Python dependencies
1. Navigate to the folder created by the cloning, using the command `cd`, followed by the directory's name.
1. Create the Conda environment for the project by typing
   ```
   conda env create -f conda-environment-files/conda-environment-cpu-<x>.yml
   ```
   , where you substitute the `<x>` by either `unix` or `win`, depending if you're following the instructions in a unix machine (Linux, Mac, etc), or a Windows one.
1. Wait for the installations to complete, and make sure there were no errors.

## Activating and testing the Conda environment
1. Now, you have created a virtual Conda environment with all Python dependencies installed.
   Still, in order to get access to this environment, you must activate it.
   This has to be done every time you start up a new terminal window.
   Activate it by the command `conda activate gnn_trajectory`.
1. You should now see `(gnn_trajectory)` printed at the start of every row in the terminal, indicating, that the `gnn_trajectory` environment is activated. If this does not happen, most probably the `conda init` step was not carried out successfully during installation.
1. Check that the installation was successful by typing
   ```
   python -c "import torch, torch_geometric, nuscenes, pyquaternion; print(torch.__version__, torch_geometric.__version__)"
   ```
   . It should print the installed versions without any errors.
1. Now you can start using Jupyter notebook. To do so, type `jupyter notebook`. Your web browser should automatically open up and navigate to http://localhost:8888, letting you access Jupyter.
1. The notebooks are available in the folder [`notebooks`](../notebooks), and are meant to be read in the order they are numbered. The preprocessed datasets are already included in `data/processed`, so `02_dataset_analysis.ipynb` and the modelling part of `03_baselines.ipynb` can be run without downloading nuScenes. The raw nuScenes data (placed in the `data` folder) is only needed for `01_nuscenes_investigate.ipynb`, the camera visualization cells of `03_baselines.ipynb` and the scripts in [`scripts`](../scripts).
1. To deactivate the conda environment, you simply type `conda deactivate`.

### Update the Conda environment
**Note:** This is important!

Make sure that your Conda environment is always up-to-date with the latest environment file available from the GitHub repo.

How to update the Conda environment:

- Navigate to the git repo you have cloned.
- Make sure that you have the latest version of the repo (see [above](#keeping-the-repo-in-sync)).
- Make sure the conda environment is deactivated (see above).
- Type e.g. `conda env update -f conda-environment-files/conda-environment-cpu-win.yml --prune`. Note however that the appropriate environment file to be used varies depending on your system. See section below for more info.
- Now you can activate the environment again.

## CPU / GPU and Windows / Unix conda environment files
On GitHub, there are different conda environment files intended for CPU / GPU as well as Windows / Unix platforms:
- `conda-environment-cpu-win.yml`
- `conda-environment-cpu-unix.yml`
- `conda-environment-gpu-win.yml`
- `conda-environment-gpu-unix.yml`

You should use the `win` rather than `unix` version if you are running Windows rather than Mac / Linux / etc. In this case, the `cpu` version should typically be used. If you have a compatible Nvidia GPU, you can however try to make use of it by simply using the `gpu` version instead, and hopefully this will work well. It can however be quite a hassle to properly install a GPU-enabled deep learning environment. Note that the `gpu` versions install PyTorch built for CUDA 11.8, and therefore cannot be used on a Mac.
