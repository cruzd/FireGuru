Hi I'm Sofia, a Neural Network model over tensorflow that makes recommendations within Vodafone's SMART Tool

1- Where is the code?
Clone me from bitbucket: https://bitbucket.corp.webdisplay.pt/projects/SMART/repos/sofia/browse

2 - How is this structured?

Real thing
| main.py - where it all starts
| smarter - where the real code is
| data - CSV files to train/test
| logs - checkpoint, model and events for inference and tensorboard
| docs - documentation

Helpers
| README.rst - Quick start guide
| LICENSE
| MANIFEST.in
| train.sh - handy shell script that trains the model
| test.sh - handy sheel script that tests inference of the model

The following are used to build wheels for other environments
| smarter.egg-info - for building the wheel to distribute to other environments
| build.sh - script that builds the smarter wheel
| setup.py
| dist - where the built wheel will be put

3 - How is inference made?
That's another project: smarterws (https://bitbucket.corp.webdisplay.pt/projects/SMART/repos/sofiaws/browse)
