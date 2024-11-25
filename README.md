# exterior-transport
This repository is meant to be an illustration on the work I conducted with M. Pegon on the exterior transport problem.

The main document is the jupyter notebook Num_tour_ext_tranps.ipynb, which gives an overview of what we implemented.

In documentation, you will find the preprint of the article explaining in great details our numerical investigation of the problem.

The folder pictures is used by the notebook to display pictures of the results of our experiments.

In src, you will find the module functions.py, which is the backbone our of python code.
The function build_source is used to generate a NxN array where the ones represent the wanted shape.
The most important function is exterior_transport, which implements a variant of the Sinkhorn algorithm
to compute the exterior transport of a given shape.
The functions below exterior_transport are used to solve our Allen-Cahn equation by alternate splitting.
Finally, the last functions are used to display matplotlib images.

See the preprint for more details.
You can also write me an email at jules.candautilh at gmail.com if you have any inquiry.

<img src="./pictures/example_pacman.png" 
     alt="A set (in blue) and its minimiser for the exterior transport (in red)." 
     style="width:30%;">


