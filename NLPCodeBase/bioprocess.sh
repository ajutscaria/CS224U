#!/bin/bash

mkdir -p state/execs
echo java -Xmx3g -cp \"classes:lib/*\" edu.stanford.nlp.bioprocess.Main -mode interactive 
java -Xmx3g -cp "classes:lib/*" edu.stanford.nlp.bioprocess.Main -mode interactive 
