SHELL := /bin/bash
TEX := lualatex
BUILD := build
POPLER_BIN := ../poster_html/tools/usr/bin
POPLER_LIB := ../poster_html/tools/usr/lib/x86_64-linux-gnu

.PHONY: all assets compile preview verify clean

all: assets compile preview verify

assets:
	python3 scripts/build_poster_assets.py

compile:
	mkdir -p $(BUILD)
	$(TEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD) poster.tex
	$(TEX) -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=$(BUILD) poster.tex
	cp $(BUILD)/poster.pdf poster.pdf

preview: compile
	PATH="$(abspath $(POPLER_BIN)):$$PATH" LD_LIBRARY_PATH="$(abspath $(POPLER_LIB)):$$LD_LIBRARY_PATH" \
		pdftoppm -f 1 -l 1 -singlefile -png -r 42 poster.pdf $(BUILD)/poster_preview
	cp $(BUILD)/poster_preview.png poster_preview.png

verify:
	python3 scripts/verify_poster.py

clean:
	rm -rf $(BUILD) poster.pdf poster_preview.png VERIFICATION.json SHA256SUMS.txt
