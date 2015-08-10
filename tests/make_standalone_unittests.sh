#!/bin/sh

g++ -std=c++11 -g  -I/home/jlovitt/git/myminerva/minerva/common unittest_fixedpoint.cpp -o unittest_fixedpoint

g++ -std=c++11 -g  -I/home/jlovitt/git/myminerva/minerva/common unittest_fixedpoint_speed.cpp -o unittest_fixedpoint_speed
