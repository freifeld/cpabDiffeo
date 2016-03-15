#!/bin/bash
#
cp test_matrix_exponential.h ./include
#
gcc -c -g -I ./include test_matrix_exponential.c >& compiler.txt
if [ $? -ne 0 ]; then
  echo "Errors compiling test_matrix_exponential.c."
  exit
fi
rm compiler.txt
#
mv test_matrix_exponential.o ./$ARCH/test_matrix_exponential.o
#
echo "Library installed as ./$ARCH/test_matrix_exponential.o"
