CUFLAGS			:= $(CUFLAGS) -std=c++11
CUFLAGSD		:= $(CUFLAGSD) -std=c++11 -g -G
OUTNAME_RELEASE = openpose
OUTNAME_DEBUG   = openpose_debug
EXTRA_DIRECTORIES = layers 
MAKEFILE ?= ./Makefile.config
SMS = 60 61 70 75
include $(MAKEFILE)
INCPATHS += -I"."

