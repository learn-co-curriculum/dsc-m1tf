# How to Install TensorFlow GPU for Mac M1/M2 with Conda

## Part I: What is the issue?

### Rosetta and emulation

You cannot run the rosetta intel emulation and the M1 chip at the same time. This [video](https://www.youtube.com/watch?v=BEUU-icPg78) by Jeff Heaton traces where the error comes from.


```
tensorflow-deps
```



Parts II, III, and IV are based on this [video](https://www.youtube.com/watch?v=5DgWvU0p2bk) by Heaton, and his GitHub repo that he refers to in the video is [here](https://github.com/jeffheaton/t81_558_deep_learning).

**Fair Warning:** *While this process has worked for me, "killing Conda" and starting fresh is naturally risky, so proceed at your own discretion.*

## Part II: Removing conda

The initial steps

1.   In the Terminal
*   >`conda activate learn-env`
*   If you already have gone through this process, instead you may have something like this instead:
> `conda activate /opt/miniconda3/envs/learn-env`
2.   Check your version
> `python --version`
3.   Install an [Anaconda](https://docs.anaconda.com/anaconda/install/uninstall/) package to git rid of the "junk" and takes care of all of the different places Anaconda is stored; type **y** and hit **return** when asked to **Proceed**.
> `conda install anaconda-clean`
4.  In the Terminal, run the package that was just downloaded, and when asked to **Delete ...**, type **y** and hit **return** each time.
> `anaconda-clean`
5.  Go to **user/** and put the **opt** or **miniconda3** folder in the trash, one can also run the code suggested on the link above for macOS.
6.  *Sanity Check.* Close the terminal, reopen, and there should be no python when you repeat Steps (1-2) from above. The terminal will return: **command not found**
*   If you are still having issues, most likely due to a previous `miniconda` install, then run the command in `base`
> `conda remove --name learn-env --all`
*   Run `conda env list` in `base` and make sure the environment you wanted to remove is gone.



## Part III: Installing and setting-up conda

The intermediate steps.

7.  You can download the full [Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Since using Miniconda gives you more control, you will download the latter. For Miniconda, there are a number of options for macOS, make sure to download the correct package. N.B. If you download the intel package, Python will still work via Rosetta emulating the intel chip, but TensorFlow will still not work since you'll be missing `tensorflow-deps`. Ergo, downloand this correct M1 package:

> **Miniconda3 macOS Apple M1 ARM 64-bit pkg**

8.  The file will be called **Miniconda3-LATEST-MacOSX-arm64.pgk**, where you know you dowloaded the correct one since you see "arm" in the name of the package. So run the installer.
*   At times, Miniconda changes the "verbiage" of the names of the packages, but they should be easily identifiable.
*   When running the install, make sure to change the location of the disk
*   Change to **Install for me only**
*   Otherwise, you'll get an error; likely something like: "*This package is incompatible with this version of macOS*"
*   You'll see me make this mistake in the video walk-thru where I change to the hard drive instead of for me only
9.  In the Terminal within **-zsh**, type `python`, this will tell you the version of Python you are running, e.g. "Python 3.10.10".
10.  *Sanity check.* Run the following two lines in z-shell, and you should get the output that confirms that you have the arm, i.e. M1/M2, version of Anaconda, and not the intel version.
```
>>> import platform
>>> platform.platform()
'macOS-13.2.1-arm64-arm-64bit'
```




## Part IV: Intalling xcode and jupyter

Here are the closing steps.

11.  Download **xcode**, or see that you already have it installed. Do this in a "fresh" terminal by running `xcode-select --install`. Either it will install it, or inform you via an error that you already have it installed.
12.  In the terminal, install Jupyter `conda install -y jupyter`, which is the main ide/editor that we'll be using. (This will take a few moments.)
13.  To move out of "(base)" within the Terminal, type `conda deactivate`. (This step may not be necessary, but does not hurt to do it.)

## Part V: Setting up the Conda Environment

This steps are layed out in more detail in this GitHub [repo](https://github.com/learn-co-curriculum/dsc-data-science-env-config).

13.  Clone the this [dsc-m1tf repo](https://github.com/learn-co-curriculum/dsc-m1tf) and cd into this repo.
14.  Create the conda environment by running the following in the Terminal, which takes a litle while since you are downloading all of the packages that you need. Most importantly for getting TensorFlow to work on the M1/M2 chip, within the **yaml** file, under *Channel* you have `apple`, under *Dependencies* you have `tensorflow-deps`, and under *pip* you have `ternsorflow-macos` and `tensorflow-metal`.
```
conda env create --name learn-env-m1tf -f mac_environment.yml
```
      *N.B.*: I had to downgrade the versions of `tensorflow-macos` and `tensorflow-metal`, viz. to 2.9 and 0.5.0, respectively, to get `model.fit()` to work correctly. Also, in the above GitHub link this is now called `mac_environment_tf.yml`.
15.  *Sanity check.* To confirm that this worked, run `conda activate learn-env-m1tf` back in "(base)". You should see that you change from "(base)" to "(learn-env-m1tf)" and then verify by running `conda info --envs`; an asterisk next to learn-env-m1tf confirms that this worked.
16.  Now set-up default environment. See the repo for `bash`, here it is for `zsh`
```
echo "conda activate learn-env" >> ~/.zshrc
source ~/.zshrc
```
16.  You now need to register the environment so that learn-env-m1tf shows up as a kernel when we run jupyter
```
conda activate learn-env-m1tf
python -m ipykernel install --user --name learn-env --display-name "Python (learn-env-m1tf)"
```




## Part VI: Testing

*Sanity check.*

Run the following code in the **learn-env kernel**.


In addition to the code all running, you hope to see for the first line of output that the Platform is "macOS-13.1-arm64-arm-64bit" and the last line is "GPU is available."


```python
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import platform

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
```

To further make sure everything is working correctly, you can run the [tutorials](https://www.tensorflow.org/tutorials) with sample code on the TensorFlow websiste.

Please let [Brendan Purdy](brendan.purdy@flatironschool.com) know via Slack if you have any questions, comments, suggestions, &c., regarding this documentation.


```python

```

IN TERMINAL (https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook)

conda deactivate
source activate learn-env-m1tf
pip install ipykernel
python -m ipykernel install --user --name learn-env-m1tf --display-name "Pyththon (m1tf)"
