CROSS_COMPILE=
CC=$(CROSS_COMPILE)gcc
CXX=$(CROSS_COMPILE)g++

INC_PATH = -I.
LIB_PATH = -L.

CFLAGS:= -g -Wall -fPIC -O2 $(INC_PATH) $(LIB_PATH)

BIN = svm_inference.so


SRCS := \
$(wildcard ./*.c)
OBJS := $(patsubst %.c,%.o,$(SRCS))

all: $(BIN)

%.o : %.c
	@echo CC $<
	@$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(OBJS)
	@echo CC $@
	@$(CC) $(CFLAGS) --shared $^ -o $@

.PHONY: clean
clean:
	@rm -rf $(OBJS) $(BIN)
