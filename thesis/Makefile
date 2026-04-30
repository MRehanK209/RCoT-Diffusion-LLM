MAIN=thesis
all:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex
clean:
	latexmk -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).bcf $(MAIN).run.xml
