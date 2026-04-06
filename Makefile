.PHONY: all build train evaluate clean

all: build train evaluate

build:
	python run_experiments.py --step build

train:
	python run_experiments.py --step train

evaluate:
	python run_experiments.py --step evaluate

clean:
	rm -rf results/graphs results/checkpoints results/figures
