This package contains the source code used in the paper
"Learning Biological Processes with Global Constraints" by
Aju Thalappillil Scaria, Jonathan Berant, Mengqiu Wang, Peter Clark, Justin Lewis, Brittany Harding and Christopher D. Manning

The quality of the code is "academic", but running it should be quite easy!

Prerequisites:
We use the optimization package Gurobi and so you need to install it (www.gurobi.com) on your machine.

Installing:
1. copy gurobi.jar from your installation to the project's lib/ directory
2. type:
    make

To run, simply type:
    ./bioprocess.sh

This will give you a commnad line interface where you can enter process descriptions and get as output
and analysis of the process.

That's it! If you want to reproduce the results from the paper you should read on.

The paper presented five systems. To run the global system type:

java -Xmx3g -cp "classes:lib/*" edu.stanford.nlp.bioprocess.Main -datasetDir lib/Dataset -execPoolDir state/execs -mode result -runOn test -runModel global

To change the system you should change the -runModel option:
For All-Prev:
    -runModel baseline
For Local_base
    -runModel localbase
For Local
    -runModel local
For Chain
    -runModel chain

Last, You can also run 10-fold cross validation for each model by chaing the -runOn option:
    -runOn dev

Be warned that running '-runOn dev -runModel global' takes a long time since there is search over the space
of coefficients for the soft constraints. 












