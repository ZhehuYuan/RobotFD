#!/usr/bin/env bash
set -euo pipefail

./run.sh 123 ../../DROID/100 > droid100
./run.sh 123 ../../DROID/100 1 > droid100FO
./run.sh 123 ../../DROID/1000 > droid1000
./run.sh 123 ../../DROID/1000 1 > droid1000FO
./run.sh 123 ../../DROID/3000 > droid3000
./run.sh 123 ../../DROID/3000 1 > droid3000FO
./run.sh 123 ../../DROID/5000 > droid5000
./run.sh 123 ../../DROID/5000 1 > droid5000FO
./run.sh 123 ../../RTK/nedLow > rtkLow
./run.sh 123 ../../RTK/nedLow 1 > rtkLowFO
./run.sh 123 ../../RTK/nedHigh > rtkHigh
./run.sh 123 ../../UZH/pure > uzh
./run.sh 123 ../../NCLT/ground_truth/ > nclt
