#!/bin/bash
#
cp c8lib.h ./include
#
gcc -c -g -I./include c8lib.c >& compiler.txt
if [ $? -ne 0 ]; then
  echo "Errors compiling c8lib.c."
  exit
fi
rm compiler.txt
#
mv c8lib.o ./$ARCH/c8lib.o
#
echo "Library installed as ./$ARCH/c8lib.o"
