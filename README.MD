## modules

Deep learning models.

+ **algorithmSim.py**: The main frame of the whole net, combining all the solitary modules. **Multichannel Gain** and the **Obstruction Network** are implemented directly in the main frame *class Algorithm*.
+ **Convolutional_RE_Net.py**: **Convolutional RE-Net** is realized.
+ **weightGen.py**: The two nonlinear models in the **Obstruction Network** and the **RM-Net** are realized here.
+ **Linear_RE_Net.py**: **Linear RE-Net** is realized.

## utils

Some useful functions and settings.

+ **Config.py**: Basic settings.
+ **dist.py**: Calculate the log-distance.
+ **line_calc.py**: Calculate the line equation.
+ **load_dict.py**: Load the trained part into the combined framework.
+ **Map.py**: Transform the location pair to a position map.

## dataset

Generate the dataset.

+ **locPair.py**: Obtain the location pair and the RSS measurements.



------



## Running Files

+ **training.py**: Train our framework.
+ **weight_pretrain**: Pre-train the nonlinear models in the **Obstruction Network** and the **RM-Net**.
+ **performance_evaluation.py**: Evaluate the performance of our framework with whole datasets.

## Tactic

+ Set the basic settings, like the dataset filename in *training.py*, all the parameters in the *Config.py*.
+ Initialize the parameters of Multichannel Gain in *algorithmSim.py*.
+ Run *weight_pretrain.py*.
+ Run *training.py*.
+ Run *performance_evaluation.py* to see the performance.