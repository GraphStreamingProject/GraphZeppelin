SHELL := /bin/bash


#  Provide a pointer to the various libraries
TOKUFTLIB = ft-index/prefix
TOKUFTLIBLINK = ../ft-index/prefix

SRCDIR = src
BUILDDIR = build

vpath %.cpp $(SRCDIR)
vpath %.h $(SRCDIR)


CORE_OBJ_O = TokuInterface.o


CPPFLAGS = -D_GNU_SOURCE -std=c++0x
CFLAGS = -g -O3 -Wall -Wextra -Werror
    
ifeq ($(USE_STATIC_LIBS),1)
LIBTOKUDB = tokufractaltree_static
LIBTOKUPORTABILITY = tokuportability_static
else
LIBTOKUDB = tokufractaltree
LIBTOKUPORTABILITY = tokuportability
endif

TOKU_LDFLAGS = -L$(TOKUFTLIB)/lib -l$(LIBTOKUDB) -l$(LIBTOKUPORTABILITY) -Wl,-rpath,$(TOKUFTLIBLINK)/lib


LDFLAGS = -lpthread -lbz2 -lz -llz4 -ldl

CPPFLAGS += -DTOKUDB -isystem$(TOKUFTLIB)/include
LDFLAGS += $(TOKU_LDFLAGS)


CORE_OBJ = $(addprefix $(BUILDDIR)/, $(CORE_OBJ_O))

OBJECTS = $(CORE_OBJ)

default: build

install: 
	. initialSetup.sh

force:
	-rm -f $(BUILDDIR)/*.o main
	+make

build: $(BUILDDIR)/main

print-%: ; @echo $*=$($*)

$(BUILDDIR)/main: $(BUILDDIR)/main.o $(OBJECTS) 
	$(CXX) $^ $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@


$(BUILDDIR)/%.o: %.cpp
	$(CXX) -c $^ $(CPPFLAGS) $(CFLAGS) -o $@

.PHONY: clean
clean:
	-rm -f $(BUILDDIR)/*.o main
