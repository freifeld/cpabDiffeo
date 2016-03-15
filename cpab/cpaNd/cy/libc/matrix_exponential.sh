#!/bin/bash
#
cp matrix_exponential.h ./include
#
gcc -c -g -I./include matrix_exponential.c >& compiler.txt
if [ $? -ne 0 ]; then
  echo "Errors compiling matrix_exponential.c."
  exit
fi
rm compiler.txt
#
mv matrix_exponential.o ./$ARCH/matrix_exponential.o
#
echo "Library installed as ./$ARCH/matrix_exponential.o"
