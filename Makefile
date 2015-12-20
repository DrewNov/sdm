COMPILER = nvcc
C_PARAMS = -x cu

SRC_DIR = src
OUT_DIR = bin
OUT_NAME = sdm

OBJS := $(addprefix $(OUT_DIR)/, main.o sdm_jaekel.o)


all: $(OUT_DIR) $(OUT_NAME)

$(OUT_DIR):
	@mkdir -p $@

$(OUT_DIR)/%.o: $(SRC_DIR)/%.c
	$(COMPILER) $(C_PARAMS) -c $< -o $@

$(OUT_NAME): $(OBJS)
	$(COMPILER) -O2 -g -pg -o $(OUT_DIR)/$@ $(OBJS)


clean:
	rm $(OUT_DIR)/*.o
