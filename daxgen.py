#!/usr/bin/env python
import os
import pwd
import sys
import time
from Pegasus.DAX3 import *

##################### BEGIN PARAMETER #####################
###################### END PARAMETER ######################

# The name of the DAX file is the first argument
if len(sys.argv) != 2:
    sys.stderr.write("Usage: %s DAXFILE\n" % (sys.argv[0]))
    sys.exit(1)
daxfile = sys.argv[1]

USER = pwd.getpwuid(os.getuid())[0]

# Create a abstract dag
print("Creating ADAG...")
dag = ADAG("fb-nlp-nmt")

# Add some workflow-level metadata
dag.metadata("creator", "%s@%s" % (USER, os.uname()[1]))
dag.metadata("created", time.ctime())

# TODO

# Write the DAX to stdout
print("Writing %s" % daxfile)
f = open(daxfile, "w")
dag.writeXML(f)
f.close()
